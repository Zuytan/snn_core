//! 📊 Standardized Neuromorphic Metrics (`no_std` pure Fix32).
//!
//! Aligned with international benchmarking criteria (**NeuroBench**, **TinyML / MLCommons**):
//! - **SynOps**: Synaptic operations distinguished into `AC` (Accumulate without multiplication) vs `MAC` (dense Multiply-Accumulate).
//! - **Sparsity**: Spatio-temporal neuronal inactivity ratio ($1 - \text{spikes}/(N \times T)$).
//! - **CPU energy estimate**: cycles × active power / clock frequency ([`EnergyEstimator::cpu_active_nanojoules`]).
//!   This is an estimate from a power figure, not a measurement; measure energy with a current probe.
//! - **Event-driven Latency**: *Latency to First Spike* & *Time-to-Decision* (TTD).
//!
//! The per-operation energy model ([`HardwareProfile`], `estimate_picojoules*`) is deprecated: its
//! constants (0.1 pJ per AC, 3.1 pJ per MAC) describe dedicated neuromorphic silicon, not a
//! microcontroller, and its Cortex-M4 constants were never measured.

// The deprecated per-operation energy model below is still implemented and tested here until it is removed.
#![allow(deprecated)]

use crate::domain::numeric::Fix32;

/// Reference hardware energy constants (CMOS 40 nm / 28 nm technology).
#[deprecated(
    since = "0.5.0",
    note = "ASIC constants, not valid for a microcontroller; use `EnergyEstimator::cpu_active_nanojoules`"
)]
pub mod energy_constants {
    /// Energy per pure accumulate (AC) event-driven synaptic operation (in picojoules: ~0.1 pJ).
    pub const ENERGY_AC_PICOJOULES_X10: u64 = 1; // 0.1 pJ represented with x10 scale

    /// Energy per dense Multiply-Accumulate (MAC) operation (in picojoules: ~3.1 pJ).
    pub const ENERGY_MAC_PICOJOULES_X10: u64 = 31; // 3.1 pJ represented with x10 scale
}

/// Differentiated synaptic operations counter (SynOps).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SynOpsTracker {
    /// Number of pure accumulation operations (AC - no multiplication, triggered by spikes).
    pub accumulate_ops: u64,
    /// Number of dense Multiply-Accumulate operations (MAC - e.g. Readout decoding layers).
    pub multiply_accumulate_ops: u64,
}

impl SynOpsTracker {
    pub const fn new() -> Self {
        Self {
            accumulate_ops: 0,
            multiply_accumulate_ops: 0,
        }
    }

    /// Records spike propagation across a given number of active synapses (fan-out).
    #[inline(always)]
    pub fn record_synaptic_spikes(&mut self, syn_count: usize) {
        self.accumulate_ops = self.accumulate_ops.saturating_add(syn_count as u64);
    }

    /// Records dense multiply-accumulate operations (e.g. linear readout).
    #[inline(always)]
    pub fn record_dense_mac(&mut self, mac_count: usize) {
        self.multiply_accumulate_ops = self
            .multiply_accumulate_ops
            .saturating_add(mac_count as u64);
    }

    /// Cumulative total of synaptic operations (SynOps = AC + MAC).
    pub const fn total_synops(&self) -> u64 {
        self.accumulate_ops
            .saturating_add(self.multiply_accumulate_ops)
    }

    /// Resets counters.
    pub fn reset(&mut self) {
        self.accumulate_ops = 0;
        self.multiply_accumulate_ops = 0;
    }
}

/// Spatio-temporal sparsity meter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SparsityMeter {
    /// Theoretical maximum firing capacity (Number of neurons * Number of ticks).
    pub total_possible_spikes: u64,
    /// Actual number of action potentials (spikes) emitted during simulation.
    pub emitted_spikes: u64,
}

impl SparsityMeter {
    pub const fn new() -> Self {
        Self {
            total_possible_spikes: 0,
            emitted_spikes: 0,
        }
    }

    /// Records time step (tick) state for a population of neurons.
    #[inline]
    pub fn record_tick(&mut self, population_size: usize, active_spikes_count: usize) {
        self.total_possible_spikes = self
            .total_possible_spikes
            .saturating_add(population_size as u64);
        self.emitted_spikes = self
            .emitted_spikes
            .saturating_add(active_spikes_count as u64);
    }

    /// Computes sparsity ratio in Fix32 fixed-point (between 0.0 and 1.0 Fix32).
    ///
    /// 1.0 = Total sparsity (complete silence).
    /// 0.0 = All neurons fire at every tick (maximum density).
    pub fn sparsity_ratio(&self) -> Fix32 {
        if self.total_possible_spikes == 0 {
            return Fix32::ONE;
        }

        let silent_spikes = self
            .total_possible_spikes
            .saturating_sub(self.emitted_spikes);
        // (silent_spikes << 16) / total_possible_spikes
        let ratio_bits = ((silent_spikes as u128) << 16) / (self.total_possible_spikes as u128);
        Fix32::from_bits(ratio_bits.min(i32::MAX as u128) as i32)
    }

    /// Computes sparsity percentage (e.g. 94.5%).
    pub fn sparsity_percent(&self) -> f32 {
        self.sparsity_ratio().to_float() * 100.0
    }

    /// Resets sparsity counters.
    pub fn reset(&mut self) {
        self.total_possible_spikes = 0;
        self.emitted_spikes = 0;
    }
}

/// Target hardware profiles for energy evaluation.
#[deprecated(
    since = "0.5.0",
    note = "per-operation constants are not measurements; use `EnergyEstimator::cpu_active_nanojoules`"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareProfile {
    /// Dedicated neuromorphic silicon accelerator CMOS 40 nm / 28 nm (Loihi, SynSense, Dynap-SE).
    /// AC ~0.1 pJ, MAC ~3.1 pJ.
    NeuromorphicAsic40nm,
    /// Von Neumann microcontroller (ARM Cortex-M0+/M4, RISC-V) with instruction cycle + SRAM access.
    /// AC ~150 pJ (~5 cycles at 30 pJ/cycle), MAC ~450 pJ (~15 cycles).
    EmbeddedMcuCortexM4,
}

/// Energy estimates. None of them is a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct EnergyEstimator;

impl EnergyEstimator {
    /// Estimates the CPU-active energy of a computation in nanojoules:
    /// `cycles × active_power / clock`.
    ///
    /// `active_power_uw` is the active power of the MCU in microwatts (for example supply voltage ×
    /// run-mode current from the datasheet, or better a value measured on the board). The result
    /// excludes sleep current, wake-ups, sensors and radio. Returns 0 when `clock_hz` is 0.
    ///
    /// ```
    /// use snn_core::metrics::EnergyEstimator;
    /// // 11,972 cycles at 48 MHz and 13.86 mW (3.3 V × 4.2 mA): about 3.46 µJ.
    /// assert_eq!(EnergyEstimator::cpu_active_nanojoules(11_972, 13_860, 48_000_000), 3_456);
    /// ```
    pub const fn cpu_active_nanojoules(cycles: u64, active_power_uw: u64, clock_hz: u64) -> u64 {
        if clock_hz == 0 {
            return 0;
        }
        // cycles / clock_hz [s] × active_power_uw [µW] = µJ; × 1000 = nJ.
        let nj = (cycles as u128)
            .saturating_mul(active_power_uw as u128)
            .saturating_mul(1000)
            / (clock_hz as u128);
        if nj > u64::MAX as u128 {
            u64::MAX
        } else {
            nj as u64
        }
    }

    /// Estimates total energy consumed by SynOps in picojoules (pJ) using default 40nm ASIC profile.
    #[deprecated(
        since = "0.5.0",
        note = "ASIC constants, not valid for a microcontroller; use `EnergyEstimator::cpu_active_nanojoules`"
    )]
    pub fn estimate_picojoules(synops: &SynOpsTracker) -> u64 {
        Self::estimate_picojoules_with_profile(synops, HardwareProfile::NeuromorphicAsic40nm)
    }

    /// Estimates total energy consumed by SynOps in picojoules (pJ) using specified hardware profile.
    #[deprecated(
        since = "0.5.0",
        note = "per-operation constants are not measurements; use `EnergyEstimator::cpu_active_nanojoules`"
    )]
    pub fn estimate_picojoules_with_profile(
        synops: &SynOpsTracker,
        profile: HardwareProfile,
    ) -> u64 {
        match profile {
            HardwareProfile::NeuromorphicAsic40nm => {
                let ac_pj_x10 = synops
                    .accumulate_ops
                    .saturating_mul(energy_constants::ENERGY_AC_PICOJOULES_X10);
                let mac_pj_x10 = synops
                    .multiply_accumulate_ops
                    .saturating_mul(energy_constants::ENERGY_MAC_PICOJOULES_X10);

                ac_pj_x10.saturating_add(mac_pj_x10) / 10
            }
            HardwareProfile::EmbeddedMcuCortexM4 => {
                // AC ~150 pJ, MAC ~450 pJ
                let ac_pj = synops.accumulate_ops.saturating_mul(150);
                let mac_pj = synops.multiply_accumulate_ops.saturating_mul(450);
                ac_pj.saturating_add(mac_pj)
            }
        }
    }

    /// Estimates total energy in nanojoules (nJ) in `Fix32` format for 40nm ASIC.
    #[deprecated(
        since = "0.5.0",
        note = "per-operation constants are not measurements; use `EnergyEstimator::cpu_active_nanojoules`"
    )]
    pub fn estimate_nanojoules_fix32(synops: &SynOpsTracker) -> Fix32 {
        Self::estimate_nanojoules_fix32_with_profile(synops, HardwareProfile::NeuromorphicAsic40nm)
    }

    /// Estimates total energy in nanojoules (nJ) in `Fix32` format using specified hardware profile.
    #[deprecated(
        since = "0.5.0",
        note = "per-operation constants are not measurements; use `EnergyEstimator::cpu_active_nanojoules`"
    )]
    pub fn estimate_nanojoules_fix32_with_profile(
        synops: &SynOpsTracker,
        profile: HardwareProfile,
    ) -> Fix32 {
        let pj = Self::estimate_picojoules_with_profile(synops, profile);
        // 1 nJ = 1000 pJ -> Fix32 = (pj * 65536) / 1000
        let nval = ((pj as u128) * 65536) / 1000;
        Fix32::from_bits(nval.min(i32::MAX as u128) as i32)
    }
}

/// Tracking event-driven inference latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct LatencyTracker {
    /// Time step where the very first network spike appeared.
    pub first_spike_tick: Option<usize>,
    /// Time step where the final decision (Readout) reached certainty criteria.
    pub decision_tick: Option<usize>,
}

impl LatencyTracker {
    pub const fn new() -> Self {
        Self {
            first_spike_tick: None,
            decision_tick: None,
        }
    }

    /// Records current tick state.
    pub fn record_tick(
        &mut self,
        tick: usize,
        active_spikes_count: usize,
        is_decision_confirmed: bool,
    ) {
        if self.first_spike_tick.is_none() && active_spikes_count > 0 {
            self.first_spike_tick = Some(tick);
        }
        if self.decision_tick.is_none() && is_decision_confirmed {
            self.decision_tick = Some(tick);
        }
    }

    /// Resets timestamps.
    pub fn reset(&mut self) {
        self.first_spike_tick = None;
        self.decision_tick = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synops_and_energy_estimation() {
        let mut tracker = SynOpsTracker::new();
        // 100 000 AC accumulations (100k * 0.1 pJ = 10 000 pJ = 10 nJ)
        tracker.record_synaptic_spikes(100_000);
        // 1 000 dense MACs (1k * 3.1 pJ = 3 100 pJ = 3.1 nJ)
        tracker.record_dense_mac(1_000);

        assert_eq!(tracker.total_synops(), 101_000);
        let energy_pj = EnergyEstimator::estimate_picojoules(&tracker);
        // 10 000 + 3 100 = 13 100 pJ
        assert_eq!(energy_pj, 13_100);

        let energy_nj = EnergyEstimator::estimate_nanojoules_fix32(&tracker);
        assert!((energy_nj.to_float() - 13.1).abs() < 0.05);

        // Realistic MCU Cortex-M4 profile
        let mcu_pj = EnergyEstimator::estimate_picojoules_with_profile(
            &tracker,
            HardwareProfile::EmbeddedMcuCortexM4,
        );
        // 100 000 * 150 + 1 000 * 450 = 15 000 000 + 450 000 = 15 450 000 pJ = 15.45 µJ
        assert_eq!(mcu_pj, 15_450_000);
    }

    #[test]
    fn test_cpu_active_energy() {
        // 48,000 cycles at 48 MHz = 1 ms; at 10 mW = 10 µJ = 10,000 nJ.
        assert_eq!(
            EnergyEstimator::cpu_active_nanojoules(48_000, 10_000, 48_000_000),
            10_000
        );
        assert_eq!(EnergyEstimator::cpu_active_nanojoules(48_000, 10_000, 0), 0);
        assert_eq!(
            EnergyEstimator::cpu_active_nanojoules(u64::MAX, u64::MAX, 1),
            u64::MAX
        );
    }

    #[test]
    fn test_sparsity_meter() {
        let mut meter = SparsityMeter::new();
        // Population of 100 neurons over 10 ticks -> 1 000 max spikes
        for _ in 0..10 {
            // Only 5 neurons fire per tick (95% silent)
            meter.record_tick(100, 5);
        }

        assert_eq!(meter.total_possible_spikes, 1000);
        assert_eq!(meter.emitted_spikes, 50);

        let ratio = meter.sparsity_ratio();
        assert_eq!(ratio, Fix32::from_num(0.95));
        assert!((meter.sparsity_percent() - 95.0).abs() < 0.01);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new();

        tracker.record_tick(0, 0, false);
        tracker.record_tick(1, 0, false);
        tracker.record_tick(2, 3, false); // First spike at tick 2
        tracker.record_tick(3, 4, true); // Decision at tick 3

        assert_eq!(tracker.first_spike_tick, Some(2));
        assert_eq!(tracker.decision_tick, Some(3));
    }
}
