//! ⚡ Silicon hardware acceleration primitives (ARM Cortex-M DSP & SIMD, RISC-V).
//!
//! Provides 1-cycle assembly optimizations and lightweight arithmetic paths
//! to maximize neuromorphic throughput on ultra-low-power microcontrollers:
//! - `ssat` : Hardware signed saturation in 1 CPU cycle (`SSAT`)
//! - `smlad` : Dual 16-bit to 32-bit multiply-accumulate in 1 CPU cycle (`SMLAD`)
//! - `fast_decay_m0` : Membrane decay without 64-bit multiplication for Cortex-M0/M0+

use crate::domain::numeric::Fix32;

/// Saturating signed addition to a 16-bit integer (range `[-32768, 32767]`).
///
/// - On ARM Cortex-M3/M4/M7/M33 with `feature = "cortex-m-dsp"`, leverages the hardware instruction `SSAT` (1 cycle).
/// - On other architectures, uses an equivalent branchless clamp.
///
/// # Example
/// ```rust
/// use snn_core::hardware::ssat_16;
///
/// assert_eq!(ssat_16(100), 100);
/// assert_eq!(ssat_16(40000), 32767);
/// assert_eq!(ssat_16(-40000), -32768);
/// ```
#[inline(always)]
pub fn ssat_16(val: i32) -> i32 {
    #[cfg(all(feature = "cortex-m-dsp", target_arch = "arm"))]
    {
        let res: i32;
        unsafe {
            core::arch::asm!(
                "ssat {out}, #16, {inp}",
                out = out(reg) res,
                inp = in(reg) val,
                options(nomem, nostack, preserves_flags)
            );
        }
        res
    }
    #[cfg(not(all(feature = "cortex-m-dsp", target_arch = "arm")))]
    {
        val.clamp(i16::MIN as i32, i16::MAX as i32)
    }
}

/// Dual 16x16-bit multiplication and accumulation into a 32-bit accumulator:
///
/// $$\text{res} = (a_{\{15:0\}} \times b_{\{15:0\}}) + (a_{\{31:16\}} \times b_{\{31:16\}}) + \text{acc}$$
///
/// - On ARM Cortex-M4/M7/M33 (`feature = "cortex-m-dsp"`), leverages the hardware instruction `SMLAD` (1 cycle).
/// - On RISC-V with `feature = "riscv-dsp"`, leverages `kmada` (RV32P draft, requires `-C target-feature=+experimental-p`).
/// - On other architectures, uses a portable saturating software fallback.
///
/// # Example
/// ```rust
/// use snn_core::hardware::smlad;
///
/// let a = (10u16 as u32) | ((20u16 as u32) << 16);
/// let b = (3u16 as u32) | ((4u16 as u32) << 16);
/// let res = smlad(a, b, 5);
/// assert_eq!(res, 115);
/// ```
#[inline(always)]
pub fn smlad(a_packed: u32, b_packed: u32, acc: i32) -> i32 {
    #[cfg(all(feature = "cortex-m-dsp", target_arch = "arm"))]
    {
        let res: i32;
        unsafe {
            core::arch::asm!(
                "smlad {out}, {a}, {b}, {acc}",
                out = out(reg) res,
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                acc = in(reg) acc,
                options(nomem, nostack, preserves_flags)
            );
        }
        res
    }
    #[cfg(all(feature = "riscv-dsp", target_arch = "riscv32"))]
    {
        let mut res = acc;
        unsafe {
            core::arch::asm!(
                "kmada {res}, {a}, {b}",
                res = inout(reg) res,
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                options(nomem, nostack, preserves_flags)
            );
        }
        res
    }
    #[cfg(not(any(
        all(feature = "cortex-m-dsp", target_arch = "arm"),
        all(feature = "riscv-dsp", target_arch = "riscv32")
    )))]
    {
        let a0 = (a_packed as i16) as i32;
        let a1 = ((a_packed >> 16) as i16) as i32;
        let b0 = (b_packed as i16) as i32;
        let b1 = ((b_packed >> 16) as i16) as i32;
        let p0 = a0 * b0;
        let p1 = a1 * b1;
        acc.saturating_add(p0).saturating_add(p1)
    }
}

/// Portable alias for `smlad` (ARM `SMLAD` / RISC-V `KMADA`).
#[inline(always)]
pub fn dual_smlad(a_packed: u32, b_packed: u32, acc: i32) -> i32 {
    smlad(a_packed, b_packed, acc)
}

/// Saturating signed addition of two pairs of 16-bit integers in 1 CPU cycle (`QADD16` / `KADD16`).
///
/// # Example
/// ```rust
/// use snn_core::hardware::qadd16;
///
/// let a = (1000u16 as u32) | ((2000u16 as u32) << 16);
/// let b = (500u16 as u32) | (((-300i16 as u16) as u32) << 16);
/// let res = qadd16(a, b);
/// assert_eq!(res as i16, 1500);
/// assert_eq!((res >> 16) as i16, 1700);
/// ```
#[inline(always)]
pub fn qadd16(a_packed: u32, b_packed: u32) -> u32 {
    #[cfg(all(feature = "cortex-m-dsp", target_arch = "arm"))]
    {
        let res: u32;
        unsafe {
            core::arch::asm!(
                "qadd16 {out}, {a}, {b}",
                out = out(reg) res,
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                options(nomem, nostack, preserves_flags)
            );
        }
        res
    }
    #[cfg(all(feature = "riscv-dsp", target_arch = "riscv32"))]
    {
        let res: u32;
        unsafe {
            core::arch::asm!(
                "kadd16 {out}, {a}, {b}",
                out = out(reg) res,
                a = in(reg) a_packed,
                b = in(reg) b_packed,
                options(nomem, nostack, preserves_flags)
            );
        }
        res
    }
    #[cfg(not(any(
        all(feature = "cortex-m-dsp", target_arch = "arm"),
        all(feature = "riscv-dsp", target_arch = "riscv32")
    )))]
    {
        let a0 = a_packed as i16;
        let a1 = (a_packed >> 16) as i16;
        let b0 = b_packed as i16;
        let b1 = (b_packed >> 16) as i16;
        let r0 = a0.saturating_add(b0) as u16 as u32;
        let r1 = a1.saturating_add(b1) as u16 as u32;
        r0 | (r1 << 16)
    }
}

/// Portable alias for `qadd16` (ARM `QADD16` / RISC-V `KADD16`).
#[inline(always)]
pub fn dual_qadd16(a_packed: u32, b_packed: u32) -> u32 {
    qadd16(a_packed, b_packed)
}

/// 4-lane 32-bit vector saturating addition.
///
/// - On ARMv8.1-M Helium (MVE) architecture with `feature = "cortex-m-helium"` and `target_feature = "mve"`,
///   directly leverages the 128-bit vector instruction `VQADD.S32`.
/// - On other architectures, uses a portable software saturating fallback.
///
/// # Example
/// ```rust
/// use snn_core::hardware::vqadd_4x32;
///
/// let a = [1000, 2000, i32::MAX - 5, i32::MIN + 5];
/// let b = [500, -300, 10, -10];
/// let res = vqadd_4x32(a, b);
/// assert_eq!(res, [1500, 1700, i32::MAX, i32::MIN]);
/// ```
#[inline(always)]
pub fn vqadd_4x32(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    #[cfg(all(
        feature = "cortex-m-helium",
        target_arch = "arm",
        target_feature = "mve"
    ))]
    {
        let mut res = [0i32; 4];
        unsafe {
            core::arch::asm!(
                "vldrw.u32 q0, [{a}]",
                "vldrw.u32 q1, [{b}]",
                "vqadd.s32 q0, q0, q1",
                "vstrw.32 q0, [{res}]",
                a = in(reg) a.as_ptr(),
                b = in(reg) b.as_ptr(),
                res = in(reg) res.as_mut_ptr(),
                out("q0") _,
                out("q1") _,
                options(nostack)
            );
        }
        res
    }
    #[cfg(not(all(
        feature = "cortex-m-helium",
        target_arch = "arm",
        target_feature = "mve"
    )))]
    {
        [
            a[0].saturating_add(b[0]),
            a[1].saturating_add(b[1]),
            a[2].saturating_add(b[2]),
            a[3].saturating_add(b[3]),
        ]
    }
}

/// 8-lane 16-bit vector saturating addition.
///
/// - On ARMv8.1-M Helium (MVE) architecture with `feature = "cortex-m-helium"` and `target_feature = "mve"`,
///   directly leverages the 128-bit vector instruction `VQADD.S16`.
/// - On other architectures, uses a portable software saturating fallback.
///
/// # Example
/// ```rust
/// use snn_core::hardware::vqadd_8x16;
///
/// let a = [10, 20, 30, 40, i16::MAX - 2, i16::MIN + 2, -100, 200];
/// let b = [1, 2, 3, 4, 10, -10, 50, -50];
/// let res = vqadd_8x16(a, b);
/// assert_eq!(res, [11, 22, 33, 44, i16::MAX, i16::MIN, -50, 150]);
/// ```
#[inline(always)]
pub fn vqadd_8x16(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
    #[cfg(all(
        feature = "cortex-m-helium",
        target_arch = "arm",
        target_feature = "mve"
    ))]
    {
        let mut res = [0i16; 8];
        unsafe {
            core::arch::asm!(
                "vldrh.u16 q0, [{a}]",
                "vldrh.u16 q1, [{b}]",
                "vqadd.s16 q0, q0, q1",
                "vstrh.16 q0, [{res}]",
                a = in(reg) a.as_ptr(),
                b = in(reg) b.as_ptr(),
                res = in(reg) res.as_mut_ptr(),
                out("q0") _,
                out("q1") _,
                options(nostack)
            );
        }
        res
    }
    #[cfg(not(all(
        feature = "cortex-m-helium",
        target_arch = "arm",
        target_feature = "mve"
    )))]
    {
        [
            a[0].saturating_add(b[0]),
            a[1].saturating_add(b[1]),
            a[2].saturating_add(b[2]),
            a[3].saturating_add(b[3]),
            a[4].saturating_add(b[4]),
            a[5].saturating_add(b[5]),
            a[6].saturating_add(b[6]),
            a[7].saturating_add(b[7]),
        ]
    }
}

/// Vector decay for 4 Fix32 membrane potentials without slow 64-bit multiplication.
#[inline(always)]
pub fn vector_decay_4x32(v_mem: [Fix32; 4], decay_q16: u32) -> [Fix32; 4] {
    [
        fast_decay_m0(v_mem[0], decay_q16),
        fast_decay_m0(v_mem[1], decay_q16),
        fast_decay_m0(v_mem[2], decay_q16),
        fast_decay_m0(v_mem[3], decay_q16),
    ]
}

/// Parallel integration and spike emission for 4 Fix32 LIF neurons.
///
/// Updates 4 neurons simultaneously: membrane decay, saturating accumulation,
/// threshold comparison, and reset.
///
/// Returns `([Fix32; 4], u8)` where `u8` is a 4-bit spike bitmask (bit $k$ set to 1 if neuron $k$ fired).
#[inline(always)]
pub fn parallel_lif_step_4x32(
    v_mem: [Fix32; 4],
    inputs: [Fix32; 4],
    decay_q16: u32,
    v_threshold: Fix32,
    v_reset: Fix32,
) -> ([Fix32; 4], u8) {
    let mut updated_v = [Fix32::ZERO; 4];
    let mut spike_mask = 0u8;

    for k in 0..4 {
        let decayed = fast_decay_m0(v_mem[k], decay_q16);
        let integrated = decayed.saturating_add(inputs[k]);

        if integrated >= v_threshold {
            updated_v[k] = v_reset;
            spike_mask |= 1 << k;
        } else {
            updated_v[k] = integrated;
        }
    }

    (updated_v, spike_mask)
}

/// Fast membrane decay multiplication for Cortex-M0/M0+ microcontrollers.
///
/// On ARMv6-M architecture (Cortex-M0 / M0+), there is no 64-bit multiplication instruction,
/// which forces the compiler to emit a slow runtime function call to `__aeabi_lmul` (~32 cycles).
/// This routine performs the multiplication $V_{\text{mem}} \times \text{decay}$ in pure 32-bit arithmetic
/// using half-word decomposition without overflow or runtime library calls.
///
/// Handles `decay_q16` up to $65536$ ($1.0$ in Q16) accurately without truncation.
///
/// # Example
/// ```rust
/// use snn_core::{Fix32, hardware::fast_decay_m0};
///
/// let v = Fix32::from_num(1.0);
/// let decayed = fast_decay_m0(v, 32768); // 0.5 factor in Q16
/// assert!((decayed.to_float() - 0.5).abs() < 0.001);
/// let no_decay = fast_decay_m0(v, 65536); // 1.0 factor in Q16
/// assert_eq!(no_decay, v);
/// ```
#[inline(always)]
pub fn fast_decay_m0(val: Fix32, decay_q16: u32) -> Fix32 {
    if decay_q16 >= 65536 {
        return val;
    }
    if decay_q16 == 0 {
        return Fix32::ZERO;
    }
    let raw = val.to_bits();
    let is_neg = raw < 0;
    let abs_val = raw.unsigned_abs();

    // Decomposition of 32-bit value: V = (V_hi << 16) + V_lo
    let v_lo = abs_val & 0xFFFF;
    let v_hi = abs_val >> 16;

    // Partial products:
    // (V_hi * decay) + ((V_lo * decay) >> 16)
    let p_hi = v_hi * decay_q16;
    let p_lo = (v_lo * decay_q16) >> 16;
    let result_mag = p_hi + p_lo;

    let res_bits = if is_neg {
        -(result_mag.min(i32::MAX as u32) as i32)
    } else {
        result_mag.min(i32::MAX as u32) as i32
    };

    Fix32::from_bits(res_bits)
}

/// Invariant integer divider without hardware division instructions.
///
/// Replaces expensive division (`/`) and modulo (`%`) operations with
/// fixed-point multiplication by the inverse $\lceil 2^{64} / D \rceil$
/// and bit shifts.
///
/// On ARM Cortex-M0/M0+ (ARMv6-M), eliminates library calls
/// to `__aeabi_uidiv` (~35-50 CPU cycles).
/// On ARM Cortex-M4/M7 (ARMv7E-M), executes in 3 clock cycles via `UMULL` (1 cycle).
///
/// # Example
/// ```rust
/// use snn_core::hardware::FastDivider;
///
/// let div = FastDivider::new(10);
/// assert_eq!(div.div(45), 4);
/// assert_eq!(div.rem(45), 5);
/// assert_eq!(div.div_rem(45), (4, 5));
/// assert_eq!(div.div_ceil(45), 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FastDivider {
    divisor: u32,
    m_hi: u32,
    m_lo: u32,
}

impl FastDivider {
    /// Precomputes magic multiplier for an invariant divisor $D$.
    ///
    /// This function is `const fn` and evaluates at compile time
    /// placing constants directly into Flash ROM (.rodata).
    #[inline(always)]
    pub const fn new(divisor: usize) -> Self {
        if divisor <= 1 {
            Self {
                divisor: divisor as u32,
                m_hi: 0,
                m_lo: 0,
            }
        } else if divisor > (u32::MAX as usize) {
            Self {
                divisor: u32::MAX,
                m_hi: 0,
                m_lo: 0,
            }
        } else {
            let d = divisor as u32;
            // M = ceil(2^64 / d)
            let m = (1u128 << 64).div_ceil(d as u128);
            Self {
                divisor: d,
                m_hi: (m >> 32) as u32,
                m_lo: (m & 0xFFFF_FFFF) as u32,
            }
        }
    }

    /// Configured divisor.
    #[inline(always)]
    pub const fn divisor(&self) -> usize {
        self.divisor as usize
    }

    /// Performs unsigned integer division $\lfloor n / D \rfloor$.
    #[inline(always)]
    pub const fn div(&self, n: usize) -> usize {
        if self.divisor <= 1 {
            return n;
        }
        if self.m_hi == 0 && self.m_lo == 0 {
            return 0;
        }
        if n > (u32::MAX as usize) {
            n / (self.divisor as usize)
        } else {
            let n32 = n as u32;
            let p_lo = (n32 as u64) * (self.m_lo as u64);
            let p_hi = (n32 as u64) * (self.m_hi as u64);
            let q = (p_hi + (p_lo >> 32)) >> 32;
            q as usize
        }
    }

    /// Computes the remainder of integer division $n \pmod D$.
    #[inline(always)]
    pub const fn rem(&self, n: usize) -> usize {
        if self.divisor <= 1 {
            0
        } else {
            let q = self.div(n);
            n - q * (self.divisor as usize)
        }
    }

    /// Computes quotient and remainder simultaneously $(q, r) = (\lfloor n / D \rfloor, n \pmod D)$.
    ///
    /// Avoids recomputing quotient during spatial index decomposition (x, y, ch).
    #[inline(always)]
    pub const fn div_rem(&self, n: usize) -> (usize, usize) {
        if self.divisor <= 1 {
            (n, 0)
        } else {
            let q = self.div(n);
            let rem = n - q * (self.divisor as usize);
            (q, rem)
        }
    }

    /// Computes division rounded up to ceiling $\lceil n / D \rceil$.
    #[inline(always)]
    pub const fn div_ceil(&self, n: usize) -> usize {
        if self.divisor <= 1 {
            return n;
        }
        if n == 0 {
            0
        } else {
            self.div(n.saturating_add((self.divisor as usize) - 1))
        }
    }
}

/// Sums four unsigned 8-bit integers packed in a 32-bit register (`USAD8` on ARMv7E-M).
///
/// $$\text{res} = \text{val}_{\{7:0\}} + \text{val}_{\{15:8\}} + \text{val}_{\{23:16\}} + \text{val}_{\{31:24\}}$$
///
/// - On ARM Cortex-M4/M7/M33 with `feature = "cortex-m-dsp"`, leverages the hardware instruction `USAD8` (1 cycle).
/// - On other architectures, uses a portable branchless shift-mask sum.
///
/// # Example
/// ```rust
/// use snn_core::hardware::hardware_usad8_sum4;
///
/// let packed = 5u32 | (10u32 << 8) | (15u32 << 16) | (20u32 << 24);
/// assert_eq!(hardware_usad8_sum4(packed), 50);
/// ```
#[inline(always)]
pub fn hardware_usad8_sum4(packed_bytes: u32) -> u32 {
    #[cfg(all(feature = "cortex-m-dsp", target_arch = "arm"))]
    {
        let res: u32;
        let zero: u32 = 0;
        unsafe {
            core::arch::asm!(
                "usad8 {out}, {inp}, {z}",
                out = out(reg) res,
                inp = in(reg) packed_bytes,
                z = in(reg) zero,
                options(nomem, nostack, preserves_flags)
            );
        }
        res
    }
    #[cfg(not(all(feature = "cortex-m-dsp", target_arch = "arm")))]
    {
        (packed_bytes & 0xFF)
            + ((packed_bytes >> 8) & 0xFF)
            + ((packed_bytes >> 16) & 0xFF)
            + ((packed_bytes >> 24) & 0xFF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssat_16_clamping() {
        assert_eq!(ssat_16(100), 100);
        assert_eq!(ssat_16(-200), -200);
        assert_eq!(ssat_16(40000), 32767);
        assert_eq!(ssat_16(-50000), -32768);
    }

    #[test]
    #[allow(clippy::decimal_bitwise_operands)]
    fn test_smlad_dual_mac() {
        // Pair a: [10, 20], Pair b: [3, 4]
        // expected: (10 * 3) + (20 * 4) + 5 = 30 + 80 + 5 = 115
        let a = (10u16 as u32) | ((20u16 as u32) << 16);
        let b = (3u16 as u32) | ((4u16 as u32) << 16);
        let acc = 5;
        let res = smlad(a, b, acc);
        assert_eq!(res, 115);

        // With negative integers
        // a: [-2, 5], b: [3, -4]
        // expected: (-2 * 3) + (5 * -4) + 10 = -6 - 20 + 10 = -16
        let a_neg = ((-2i16 as u16) as u32) | (((5i16 as u16) as u32) << 16);
        let b_neg = ((3i16 as u16) as u32) | (((-4i16 as u16) as u32) << 16);
        let res_neg = smlad(a_neg, b_neg, 10);
        assert_eq!(res_neg, -16);
    }

    #[test]
    fn test_fast_decay_m0_accuracy() {
        // Decay factor 0.9 in Q16 = round(0.9 * 65536) = 58982
        let decay_factor = 58982u32;
        let v = Fix32::from_num(1.5);
        let fast_res = fast_decay_m0(v, decay_factor);

        // Comparison with standard 64-bit arithmetic
        let std_res = v * Fix32::from_bits(decay_factor as i32);
        let diff = (fast_res.to_bits() - std_res.to_bits()).abs();
        assert!(diff <= 1, "Difference {} exceeds 1 LSB", diff);

        // Decay factor 1.0 in Q16 = 65536 (must preserve potential exactly)
        let no_decay = fast_decay_m0(v, 65536);
        assert_eq!(no_decay, v);

        // Negative potential with decay
        let v_neg = Fix32::from_num(-1.5);
        let fast_neg = fast_decay_m0(v_neg, decay_factor);
        let std_neg = v_neg * Fix32::from_bits(decay_factor as i32);
        let diff_neg = (fast_neg.to_bits() - std_neg.to_bits()).abs();
        assert!(diff_neg <= 1, "Difference {} exceeds 1 LSB", diff_neg);

        // Zero decay factor
        let zero_res = fast_decay_m0(v, 0);
        assert_eq!(zero_res, Fix32::ZERO);
    }

    #[test]
    #[allow(clippy::decimal_bitwise_operands)]
    fn test_qadd16_dual_addition() {
        // Pair a: [1000, 2000], Pair b: [500, -300]
        // Expected result: [1500, 1700]
        let a = (1000u16 as u32) | ((2000u16 as u32) << 16);
        let b = (500u16 as u32) | (((-300i16 as u16) as u32) << 16);
        let res = qadd16(a, b);
        let r0 = res as i16;
        let r1 = (res >> 16) as i16;
        assert_eq!(r0, 1500);
        assert_eq!(r1, 1700);

        // Test with saturation
        // a: [32000, -32000], b: [2000, -2000]
        let a_sat = (32000u16 as u32) | (((-32000i16 as u16) as u32) << 16);
        let b_sat = (2000u16 as u32) | (((-2000i16 as u16) as u32) << 16);
        let res_sat = qadd16(a_sat, b_sat);
        assert_eq!(res_sat as i16, 32767);
        assert_eq!((res_sat >> 16) as i16, -32768);
    }

    #[test]
    fn test_fast_divider_parity_and_operations() {
        let divisors = [
            1, 2, 3, 4, 5, 7, 8, 10, 16, 28, 32, 64, 100, 128, 256, 1000, 65535,
        ];
        let test_values = [
            0, 1, 2, 3, 5, 10, 27, 28, 29, 99, 100, 101, 1000, 65534, 65535, 65536, 100000,
        ];

        for &d in &divisors {
            let fd = FastDivider::new(d);
            assert_eq!(fd.divisor(), d);

            for &n in &test_values {
                let expected_div = n / d;
                let expected_rem = n % d;
                let expected_ceil = n.div_ceil(d);

                assert_eq!(fd.div(n), expected_div, "div failed for n={}, d={}", n, d);
                assert_eq!(fd.rem(n), expected_rem, "rem failed for n={}, d={}", n, d);
                assert_eq!(
                    fd.div_rem(n),
                    (expected_div, expected_rem),
                    "div_rem failed for n={}, d={}",
                    n,
                    d
                );
                assert_eq!(
                    fd.div_ceil(n),
                    expected_ceil,
                    "div_ceil failed for n={}, d={}",
                    n,
                    d
                );
            }
        }
    }

    #[test]
    fn test_helium_vector_saturating_adds() {
        let a4 = [1000, 2000, i32::MAX - 5, i32::MIN + 5];
        let b4 = [500, -300, 10, -10];
        let r4 = vqadd_4x32(a4, b4);
        assert_eq!(r4, [1500, 1700, i32::MAX, i32::MIN]);

        let a8 = [10, 20, 30, 40, i16::MAX - 2, i16::MIN + 2, -100, 200];
        let b8 = [1, 2, 3, 4, 10, -10, 50, -50];
        let r8 = vqadd_8x16(a8, b8);
        assert_eq!(r8, [11, 22, 33, 44, i16::MAX, i16::MIN, -50, 150]);
    }

    #[test]
    fn test_parallel_lif_step_4x32() {
        // decay_q16: 0.5 in Q16 = 32768
        let decay = 32768u32;
        let v_threshold = Fix32::from_num(1.0);
        let v_reset = Fix32::ZERO;

        // Neuron 0: 0.6 * 0.5 + 0.8 = 0.3 + 0.8 = 1.1 >= 1.0 -> SPIKE (v = 0.0)
        // Neuron 1: 0.4 * 0.5 + 0.1 = 0.2 + 0.1 = 0.3 < 1.0  -> NO SPIKE (v = 0.3)
        // Neuron 2: 0.0 * 0.5 + 1.2 = 1.2 >= 1.0              -> SPIKE (v = 0.0)
        // Neuron 3: 0.8 * 0.5 + 0.0 = 0.4 < 1.0              -> NO SPIKE (v = 0.4)
        let v_mem = [
            Fix32::from_num(0.6),
            Fix32::from_num(0.4),
            Fix32::ZERO,
            Fix32::from_num(0.8),
        ];
        let inputs = [
            Fix32::from_num(0.8),
            Fix32::from_num(0.1),
            Fix32::from_num(1.2),
            Fix32::ZERO,
        ];

        let (new_v, spike_mask) =
            parallel_lif_step_4x32(v_mem, inputs, decay, v_threshold, v_reset);

        // Mask: neuron 0 (bit 0 = 1) and neuron 2 (bit 2 = 4) -> mask = 1 | 4 = 5 (0b0101)
        assert_eq!(spike_mask, 0b0101);
        assert_eq!(new_v[0], v_reset);
        assert_eq!(new_v[2], v_reset);
        assert!((new_v[1].to_float() - 0.3).abs() < 0.01);
        assert!((new_v[3].to_float() - 0.4).abs() < 0.01);
    }

    #[test]
    #[allow(clippy::decimal_bitwise_operands)]
    fn test_hardware_usad8_sum4() {
        let packed = 12u32 | (34u32 << 8) | (56u32 << 16) | (78u32 << 24);
        assert_eq!(hardware_usad8_sum4(packed), 12 + 34 + 56 + 78);

        let max_packed = 0xFFFF_FFFFu32;
        assert_eq!(hardware_usad8_sum4(max_packed), 255 * 4);

        let zero_packed = 0u32;
        assert_eq!(hardware_usad8_sum4(zero_packed), 0);
    }
}
