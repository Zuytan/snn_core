use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 32-bit fixed-point type (Q16.16) optimized for FPU-less processors (`no_std`).
///
/// The 16 most significant bits represent the integer part (-32,768 to +32,767),
/// and the 16 least significant bits represent the fractional part (resolution ~0.000015).
///
/// Arithmetic saturates instead of wrapping. Multiplication rounds half up (toward +infinity),
/// which is a small positive bias of at most half an LSB per operation, not round-half-even.
///
/// # Example
/// ```rust
/// use snn_core::Fix32;
///
/// let a = Fix32::from_num(1.5);
/// let b = Fix32::from_num(2.0);
/// let sum = a + b;
/// assert_eq!(sum, Fix32::from_num(3.5));
///
/// let product = a * b;
/// assert_eq!(product, Fix32::from_num(3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct Fix32(pub i32);

impl Fix32 {
    pub const FRACTIONAL_BITS: u32 = 16;
    pub const SCALE: i32 = 1 << 16; // 65536
    pub const ZERO: Self = Fix32(0);
    pub const ONE: Self = Fix32(1 << 16);
    pub const MAX: Self = Fix32(i32::MAX);
    pub const MIN: Self = Fix32(i32::MIN);

    #[inline]
    pub const fn from_bits(bits: i32) -> Self {
        Fix32(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> i32 {
        self.0
    }

    #[inline]
    pub const fn from_int(val: i32) -> Self {
        if val > 32767 {
            Fix32::MAX
        } else if val < -32768 {
            Fix32::MIN
        } else {
            Fix32(val << 16)
        }
    }

    #[inline]
    pub fn from_num(val: f32) -> Self {
        if val.is_nan() {
            return Fix32::ZERO;
        }
        let clamped = val.clamp(-32768.0, 32767.99998);
        Fix32((clamped * 65536.0) as i32)
    }

    #[inline]
    pub fn to_float(self) -> f32 {
        self.0 as f32 / 65536.0
    }

    #[inline]
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Fix32(self.0.saturating_add(rhs.0))
    }

    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Fix32(self.0.saturating_sub(rhs.0))
    }

    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Fix32(self.0.saturating_abs())
    }

    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.0 == 0 {
            None
        } else {
            let numer = i64::from(self.0) << 16;
            let denom = i64::from(rhs.0);
            let res = numer / denom;
            if res > i64::from(i32::MAX) || res < i64::from(i32::MIN) {
                None
            } else {
                Some(Fix32(res as i32))
            }
        }
    }
}

impl Add for Fix32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Fix32(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for Fix32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl Sub for Fix32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Fix32(self.0.saturating_sub(rhs.0))
    }
}

impl SubAssign for Fix32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_sub(rhs.0);
    }
}

impl Mul for Fix32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let product = i64::from(self.0) * i64::from(rhs.0);
        let res = (product + (1 << 15)) >> 16;
        if res > i64::from(i32::MAX) {
            Fix32::MAX
        } else if res < i64::from(i32::MIN) {
            Fix32::MIN
        } else {
            Fix32(res as i32)
        }
    }
}

impl MulAssign for Fix32 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Fix32 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.0 == 0 {
            if self.0 > 0 {
                return Fix32::MAX;
            } else if self.0 < 0 {
                return Fix32::MIN;
            }
            return Fix32::ZERO;
        }
        let numer = i64::from(self.0) << 16;
        let denom = i64::from(rhs.0);
        let res = numer / denom;
        if res > i64::from(i32::MAX) {
            Fix32::MAX
        } else if res < i64::from(i32::MIN) {
            Fix32::MIN
        } else {
            Fix32(res as i32)
        }
    }
}

impl DivAssign for Fix32 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Fix32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Fix32(self.0.saturating_neg())
    }
}

impl Sum for Fix32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fix32::ZERO, |a, b| a + b)
    }
}

/// A 16-bit fixed-point type (Q8.8) optimized for ultra-lightweight FPU-less processors (Cortex-M0+, RISC-V).
///
/// The 8 most significant bits represent the integer part (-128 to +127),
/// and the 8 least significant bits represent the fractional part (resolution ~0.0039).
/// Multiplication operations execute in 1 machine cycle on 32-bit registers (MULS)
/// without 64-bit emulation or SRAM overhead.
///
/// # Example
/// ```rust
/// use snn_core::Fix16;
///
/// let a = Fix16::from_num(1.5);
/// let b = Fix16::from_num(2.0);
/// let sum = a + b;
/// assert_eq!(sum, Fix16::from_num(3.5));
///
/// let product = a * b;
/// assert_eq!(product, Fix16::from_num(3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Fix16(pub i16);

impl Fix16 {
    pub const FRACTIONAL_BITS: u32 = 8;
    pub const SCALE: i16 = 1 << 8; // 256
    pub const ZERO: Self = Fix16(0);
    pub const ONE: Self = Fix16(1 << 8);
    pub const MAX: Self = Fix16(i16::MAX);
    pub const MIN: Self = Fix16(i16::MIN);

    #[inline]
    pub const fn from_bits(bits: i16) -> Self {
        Fix16(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> i16 {
        self.0
    }

    #[inline]
    pub const fn from_int(val: i16) -> Self {
        Fix16(val.saturating_mul(256))
    }

    #[inline]
    pub fn from_num(val: f32) -> Self {
        Fix16((val * 256.0).clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16)
    }

    #[inline]
    pub fn to_float(self) -> f32 {
        f32::from(self.0) / 256.0
    }

    #[inline]
    pub const fn to_fix32(self) -> Fix32 {
        Fix32::from_bits((self.0 as i32) << 8)
    }

    #[inline]
    pub const fn from_fix32(val: Fix32) -> Self {
        let shifted = val.0 >> 8;
        if shifted > i16::MAX as i32 {
            Fix16::MAX
        } else if shifted < i16::MIN as i32 {
            Fix16::MIN
        } else {
            Fix16(shifted as i16)
        }
    }

    #[inline]
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Fix16(self.0.saturating_add(rhs.0))
    }

    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Fix16(self.0.saturating_sub(rhs.0))
    }

    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Fix16(self.0.saturating_abs())
    }
}

impl Add for Fix16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Fix16(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for Fix16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl Sub for Fix16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Fix16(self.0.saturating_sub(rhs.0))
    }
}

impl SubAssign for Fix16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_sub(rhs.0);
    }
}

impl Mul for Fix16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let product = i32::from(self.0) * i32::from(rhs.0);
        let res = (product + (1 << 7)) >> 8;
        if res > i32::from(i16::MAX) {
            Fix16::MAX
        } else if res < i32::from(i16::MIN) {
            Fix16::MIN
        } else {
            Fix16(res as i16)
        }
    }
}

impl MulAssign for Fix16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Fix16 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.0 == 0 {
            if self.0 > 0 {
                return Fix16::MAX;
            } else if self.0 < 0 {
                return Fix16::MIN;
            }
            return Fix16::ZERO;
        }
        let numer = i32::from(self.0) << 8;
        let denom = i32::from(rhs.0);
        let res = numer / denom;
        if res > i32::from(i16::MAX) {
            Fix16::MAX
        } else if res < i32::from(i16::MIN) {
            Fix16::MIN
        } else {
            Fix16(res as i16)
        }
    }
}

impl DivAssign for Fix16 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Fix16 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Fix16(self.0.saturating_neg())
    }
}

impl Sum for Fix16 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fix16::ZERO, |a, b| a + b)
    }
}

/// Unifying trait for fixed-point and integer scalar types (`Fix32`, `Fix16`).
/// Enables configuring neuromorphic networks for ultra-low memory (Fix16 Q8.8, 2 bytes per state)
/// or high standard precision (Fix32 Q16.16, 4 bytes per state).
pub trait NumericScalar:
    Copy
    + Default
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + core::fmt::Debug
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
    + Sum
{
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
    const MIN: Self;

    fn from_num(val: f32) -> Self;
    fn to_float(self) -> f32;
    fn from_fix32(val: Fix32) -> Self;
    #[must_use]
    fn saturating_add(self, rhs: Self) -> Self;
    #[must_use]
    fn saturating_sub(self, rhs: Self) -> Self;
    #[must_use]
    fn abs(self) -> Self;
    fn to_fix32(self) -> Fix32;
}

impl NumericScalar for Fix32 {
    const ZERO: Self = Fix32::ZERO;
    const ONE: Self = Fix32::ONE;
    const MAX: Self = Fix32::MAX;
    const MIN: Self = Fix32::MIN;

    #[inline(always)]
    fn from_num(val: f32) -> Self {
        Fix32::from_num(val)
    }

    #[inline(always)]
    fn to_float(self) -> f32 {
        self.to_float()
    }

    #[inline(always)]
    fn from_fix32(val: Fix32) -> Self {
        val
    }

    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }

    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn to_fix32(self) -> Fix32 {
        self
    }
}

impl NumericScalar for Fix16 {
    const ZERO: Self = Fix16::ZERO;
    const ONE: Self = Fix16::ONE;
    const MAX: Self = Fix16::MAX;
    const MIN: Self = Fix16::MIN;

    #[inline(always)]
    fn from_num(val: f32) -> Self {
        Fix16::from_num(val)
    }

    #[inline(always)]
    fn to_float(self) -> f32 {
        self.to_float()
    }

    #[inline(always)]
    fn from_fix32(val: Fix32) -> Self {
        Fix16::from_fix32(val)
    }

    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }

    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn to_fix32(self) -> Fix32 {
        self.to_fix32()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_fix32_conversions() {
        let one = Fix32::from_int(1);
        assert_eq!(one.0, 65536);
        assert_eq!(one.to_float(), 1.0);

        let half = Fix32::from_num(0.5);
        assert_eq!(half.0, 32768);
        assert_eq!(half.to_float(), 0.5);
    }

    #[test]
    fn test_fix32_arithmetic() {
        let a = Fix32::from_num(1.5);
        let b = Fix32::from_num(2.0);

        assert_eq!((a + b).to_float(), 3.5);
        assert_eq!((b - a).to_float(), 0.5);
        assert_eq!((a * b).to_float(), 3.0);
        assert_eq!((b / a).to_float(), 1.3333282); // Q16.16 approximation
    }

    #[test]
    fn test_fix32_div_by_zero_saturates() {
        let pos = Fix32::from_int(10);
        let neg = Fix32::from_int(-10);
        let zero = Fix32::ZERO;

        assert_eq!(pos / zero, Fix32::MAX);
        assert_eq!(neg / zero, Fix32::MIN);
        assert_eq!(zero / zero, Fix32::ZERO);
    }

    #[test]
    fn test_fix32_overflow_saturates() {
        let big = Fix32::MAX;
        let tiny = Fix32::from_bits(1); // 1/65536
        assert_eq!(big / tiny, Fix32::MAX);
    }

    #[test]
    fn test_fix32_saturating_add_sub() {
        let max = Fix32::MAX;
        let min = Fix32::MIN;
        let one = Fix32::ONE;

        assert_eq!(max.saturating_add(one), Fix32::MAX);
        assert_eq!(min.saturating_sub(one), Fix32::MIN);
        // Test standard operators + and - saturation
        assert_eq!(max + one, Fix32::MAX);
        assert_eq!(min - one, Fix32::MIN);
        // Test * operator saturation
        assert_eq!(max * Fix32::from_int(2), Fix32::MAX);
        assert_eq!(min * Fix32::from_int(2), Fix32::MIN);
        // Test neg saturation on MIN
        assert_eq!(-min, Fix32::MAX);
    }

    #[test]
    fn test_fix16_arithmetic_and_conversions() {
        let one = Fix16::from_int(1);
        assert_eq!(one.0, 256);
        assert_eq!(one.to_float(), 1.0);

        let half = Fix16::from_num(0.5);
        assert_eq!(half.0, 128);
        assert_eq!(half.to_float(), 0.5);

        let a = Fix16::from_num(1.5);
        let b = Fix16::from_num(2.0);
        assert_eq!((a + b).to_float(), 3.5);
        assert_eq!((b - a).to_float(), 0.5);
        assert_eq!((a * b).to_float(), 3.0);
        assert_eq!((b / a).to_float(), 1.3320312); // Q8.8 approximation

        // Conversion Fix16 <-> Fix32
        let f32_val = a.to_fix32();
        assert_eq!(f32_val.to_float(), 1.5);
        let back = Fix16::from_fix32(f32_val);
        assert_eq!(back, a);

        // Saturation
        assert_eq!(Fix16::MAX + Fix16::ONE, Fix16::MAX);
        assert_eq!(Fix16::MIN - Fix16::ONE, Fix16::MIN);
        assert_eq!(Fix16::from_int(10) / Fix16::ZERO, Fix16::MAX);
    }
}
