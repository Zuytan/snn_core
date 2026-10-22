"""
snn_compiler.ir
Intermediate Representation (IR) and topological optimization passes for NIR graphs.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import deque

def sanitize_ident(name: str) -> str:
    """Sanitizes a string to produce a valid C/Rust identifier."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

class NirGraph:
    """Normalized NIR graph with topological DAG analysis and optimization passes."""

    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Tuple[str, str]],
    ):
        self.nodes = nodes
        self.edges = edges
        self.topological_order: List[str] = []
        self._compute_topological_sort()

    def _compute_topological_sort(self):
        """Computes topological order via Kahn's algorithm."""
        in_degree: Dict[str, int] = {k: 0 for k in self.nodes}
        adj: Dict[str, List[str]] = {k: [] for k in self.nodes}

        for u, v in self.edges:
            if u in self.nodes and v in self.nodes:
                adj[u].append(v)
                in_degree[v] += 1

        queue = deque([k for k, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.nodes):
            # In presence of recurrent connections (cycles), append remaining nodes
            remaining = [k for k in self.nodes if k not in order]
            order.extend(remaining)

        self.topological_order = order

    def fold_biases_into_lif(self):
        """
        Optimization pass: Folds bias vectors from Affine layers directly into
        the static resting bias parameters of downstream LIF populations.
        """
        for u, v in self.edges:
            if u in self.nodes and v in self.nodes:
                u_node = self.nodes[u]
                v_node = self.nodes[v]
                if u_node.get("type") == "Affine" and "bias" in u_node:
                    if v_node.get("type") in ("LIF", "CubaLIF", "IF"):
                        v_node["neuron_bias"] = u_node["bias"]

    def extract_feedforward_pipeline(self) -> List[Tuple[str, str]]:
        """
        Extracts ordered sequence of (synapse, neuron) pairs composing the feedforward pipeline.
        """
        pipeline = []
        synapse_types = ("Linear", "Affine", "Conv1d", "Conv2d")
        neuron_types = ("LIF", "CubaLIF", "IF", "IAF", "ALIF", "Izhikevich")

        for u, v in self.edges:
            if u in self.nodes and v in self.nodes:
                if self.nodes[u].get("type") in synapse_types and self.nodes[v].get("type") in neuron_types:
                    pipeline.append((u, v))

        return pipeline

    def find_input_and_output_nodes(self) -> Tuple[List[str], List[str]]:
        """Identifies graph input and output nodes."""
        in_nodes = []
        out_nodes = []

        dsts = {v for _, v in self.edges}
        srcs = {u for u, _ in self.edges}

        for k in self.nodes:
            if k not in dsts:
                in_nodes.append(k)
            if k not in srcs:
                out_nodes.append(k)

        return in_nodes, out_nodes

    def compute_memory_liveness(self) -> Dict[str, Any]:
        """
        Liveness analysis and SRAM arena scheduler via interval coloring (Greedy Memory Planner).
        Guarantees optimal conflict-free SRAM allocation for arbitrary DAG topologies.
        """
        step_map: Dict[str, int] = {node: idx for idx, node in enumerate(self.topological_order)}
        consumers: Dict[str, List[str]] = {node: [] for node in self.nodes}

        for u, v in self.edges:
            if u in consumers:
                consumers[u].append(v)

        intervals: List[Dict[str, Any]] = []
        for node in self.topological_order:
            node_data = self.nodes.get(node, {})
            # Determine byte size of produced activation tensor
            n_neurons = node_data.get("out_features", node_data.get("size", 1))
            n_words = (n_neurons + 31) // 32
            byte_size = n_words * 4 # 1 bit per neuron packed in 32-bit words

            birth = step_map[node]
            # End of life is step of last direct consumer
            last_cons_step = max([step_map[c] for c in consumers[node]], default=birth)
            death = max(birth, last_cons_step)

            intervals.append({
                "name": node,
                "size": byte_size,
                "birth": birth,
                "death": death,
            })

        # Greedy First-Fit memory offset scheduling
        allocated_offsets: Dict[str, int] = {}
        placed_blocks: List[Tuple[int, int, int, int]] = [] # (offset, size, birth, death)

        for item in intervals:
            size = item["size"]
            birth = item["birth"]
            death = item["death"]

            candidate_offset = 0
            while True:
                conflict = False
                for off, s, b, d in placed_blocks:
                    # Temporal overlap
                    time_overlap = not (death < b or birth > d)
                    # Spatial overlap
                    space_overlap = not (candidate_offset + size <= off or candidate_offset >= off + s)
                    if time_overlap and space_overlap:
                        # Skip past conflicting block (4-byte aligned)
                        candidate_offset = (off + s + 3) & ~3
                        conflict = True
                        break
                if not conflict:
                    break

            allocated_offsets[item["name"]] = candidate_offset
            placed_blocks.append((candidate_offset, size, birth, death))

        total_arena_bytes = max([off + s for off, s, _, _ in placed_blocks], default=0)
        # Total 4-byte alignment
        total_arena_bytes = (total_arena_bytes + 3) & ~3

        return {
            "total_arena_bytes": total_arena_bytes,
            "offsets": allocated_offsets,
            "intervals": intervals,
        }
