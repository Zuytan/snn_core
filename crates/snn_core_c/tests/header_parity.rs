//! Guards `include/snn_core.h` against drifting from the Rust FFI (the Rust code is the source of truth).
//!
//! Checks that every exported `snn_*` function is declared in the header with the same number of
//! parameters, and that every `#[repr(C)]` struct in `ffi.rs` has the same ordered field names in the header.

use std::collections::BTreeMap;

fn read(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn strip_c_comments(s: &str) -> String {
    let mut out = String::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                i += 1;
            }
            i += 2;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn count_params(list: &str) -> usize {
    let t = list.trim();
    if t.is_empty() || t == "void" {
        0
    } else {
        t.split(',').count()
    }
}

/// name -> parameter count of the `extern "C"` functions in ffi.rs.
fn rust_functions(src: &str) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    let mut rest = src;
    while let Some(pos) = rest.find("extern \"C\" fn ") {
        rest = &rest[pos + "extern \"C\" fn ".len()..];
        let name_end = rest.find('(').unwrap();
        let name = rest[..name_end].trim().to_string();
        let mut depth = 0;
        let mut end = name_end;
        for (i, c) in rest[name_end..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = name_end + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let params: Vec<&str> = rest[name_end + 1..end]
            .split(',')
            .filter(|p| !p.trim().is_empty())
            .collect();
        if name.starts_with("snn_") {
            map.insert(name, params.len());
        }
    }
    map
}

fn header_functions(h: &str) -> BTreeMap<String, usize> {
    let mut in_macro = false;
    let mut kept = Vec::new();
    let stripped = strip_c_comments(h);
    for l in stripped.lines() {
        let is_pp = l.trim_start().starts_with('#') || in_macro;
        in_macro = is_pp && l.trim_end().ends_with('\\');
        if !is_pp && l.trim() != "}" && !l.contains("extern \"C\" {") {
            kept.push(l);
        }
    }
    let h = kept.join("\n");
    let mut map = BTreeMap::new();
    for stmt in h.split(';') {
        let Some(open) = stmt.find('(') else { continue };
        if stmt.contains('{') || stmt.contains("typedef") {
            continue;
        }
        let head = &stmt[..open];
        let name = head
            .split_whitespace()
            .last()
            .unwrap_or("")
            .trim_start_matches('*');
        if !name.starts_with("snn_") {
            continue;
        }
        let close = stmt.rfind(')').unwrap();
        map.insert(name.to_string(), count_params(&stmt[open + 1..close]));
    }
    map
}

fn rust_structs(src: &str) -> BTreeMap<String, Vec<String>> {
    let mut map = BTreeMap::new();
    let mut rest = src;
    while let Some(pos) = rest.find("#[repr(C)]") {
        rest = &rest[pos..];
        let Some(spos) = rest.find("pub struct ") else {
            break;
        };
        let eol = rest.find("\npub enum ").unwrap_or(usize::MAX);
        if eol < spos {
            rest = &rest[eol + 1..];
            continue;
        }
        let after = &rest[spos + "pub struct ".len()..];
        let name_end = after
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap();
        let name = after[..name_end].to_string();
        let body_start = after.find('{').unwrap();
        let body_end = after.find("\n}").unwrap();
        let fields = after[body_start + 1..body_end]
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.strip_prefix("pub ")
                    .and_then(|r| r.split(':').next())
                    .map(|n| n.trim().to_string())
            })
            .collect();
        map.insert(name, fields);
        rest = &after[body_end..];
    }
    map
}

fn header_structs(h: &str) -> BTreeMap<String, Vec<String>> {
    let h = strip_c_comments(h);
    let mut map = BTreeMap::new();
    let mut rest = h.as_str();
    while let Some(pos) = rest.find("typedef struct {") {
        let after = &rest[pos + "typedef struct {".len()..];
        let close = after.find('}').unwrap();
        let body = &after[..close];
        let tail = &after[close + 1..];
        let name = tail[..tail.find(';').unwrap()].trim().to_string();
        let fields = body
            .split(';')
            .filter_map(|f| {
                let f = f.trim();
                (!f.is_empty()).then(|| {
                    f.rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap()
                        .to_string()
                })
            })
            .collect();
        map.insert(name, fields);
        rest = tail;
    }
    map
}

#[test]
fn every_exported_function_is_declared_with_same_arity() {
    let rs = rust_functions(&read("src/ffi.rs"));
    let h = header_functions(&read("../../include/snn_core.h"));
    assert!(
        rs.len() > 20,
        "parser found too few Rust exports: {}",
        rs.len()
    );
    for (name, n) in &rs {
        match h.get(name) {
            None => panic!("`{name}` is exported by Rust but not declared in include/snn_core.h"),
            Some(m) => assert_eq!(n, m, "`{name}`: {n} params in Rust, {m} in the header"),
        }
    }
    for name in h.keys() {
        assert!(
            rs.contains_key(name),
            "`{name}` is declared in the header but not exported by Rust"
        );
    }
}

#[test]
fn repr_c_structs_match_header_field_order() {
    let rs = rust_structs(&read("src/ffi.rs"));
    let h = header_structs(&read("../../include/snn_core.h"));
    assert!(
        rs.len() >= 8,
        "parser found too few Rust structs: {}",
        rs.len()
    );
    for (name, fields) in &rs {
        match h.get(name) {
            None => {
                panic!("struct `{name}` is #[repr(C)] in Rust but missing in include/snn_core.h")
            }
            Some(hf) => assert_eq!(
                fields, hf,
                "struct `{name}`: field list differs between Rust and header"
            ),
        }
    }
}
