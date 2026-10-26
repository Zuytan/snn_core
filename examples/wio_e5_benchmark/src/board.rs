//! Board support shared by every firmware binary: system clock and cycle timing.
//!
//! After reset the STM32WLE5 runs from MSI at 4 MHz with 0 Flash wait states. The benchmarks
//! are reported at 48 MHz, so [`init_clock_48mhz`] switches MSI to range 11 (48 MHz) after
//! raising the Flash latency to 2 wait states (RM0461, voltage range 1). [`hclk_hz`] reads the
//! clock back from the RCC registers, so printed times use the frequency the core really runs at.
//! Register addresses and fields: RM0461 (STM32WLEx reference manual).

// Each binary includes this module and uses only part of it.
#![allow(dead_code)]

use core::ptr::{read_volatile, write_volatile};

const RCC_CR: *mut u32 = 0x5800_0000 as *mut u32;
const RCC_CFGR: *const u32 = 0x5800_0008 as *const u32;
const RCC_CSR: *const u32 = 0x5800_0094 as *const u32;
const FLASH_ACR: *mut u32 = 0x5800_4000 as *mut u32;
const PWR_CR1: *const u32 = 0x5800_0400 as *const u32;

const RCC_CR_MSIRDY: u32 = 1 << 1;
const RCC_CR_MSIRGSEL: u32 = 1 << 3;
const RCC_CR_MSIRANGE_MASK: u32 = 0xF << 4;
const MSI_RANGE_48MHZ: u32 = 11;
const FLASH_ACR_LATENCY_MASK: u32 = 0x7;
const FLASH_LATENCY_48MHZ: u32 = 2;

/// MSI frequency of each MSIRANGE value (RM0461, RCC_CR.MSIRANGE).
const MSI_RANGE_HZ: [u32; 12] = [
    100_000, 200_000, 400_000, 800_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000,
    24_000_000, 32_000_000, 48_000_000,
];

/// Run-mode current used for the CPU-active energy estimate, in microamperes.
///
/// This 4.2 mA figure at 48 MHz has been used since the first report; it was not measured on this
/// board. Replace it with a current-probe measurement (phase 2 of the plan).
pub const ASSUMED_RUN_CURRENT_UA: u64 = 4_200;
/// Supply voltage used for the CPU-active energy estimate, in millivolts.
pub const ASSUMED_SUPPLY_MV: u64 = 3_300;

/// Switches the system clock to MSI 48 MHz with 2 Flash wait states.
///
/// Returns `false`, leaving the clock unchanged, if the regulator is not in voltage range 1
/// (48 MHz is not allowed in range 2).
pub fn init_clock_48mhz() -> bool {
    // SAFETY: fixed, aligned addresses of memory-mapped STM32WL registers (RM0461). Called once at
    // startup by a single thread with no interrupt touching RCC, FLASH or PWR.
    unsafe {
        // PWR_CR1.VOS (bits 10:9) = 0b01: range 1, the reset value.
        if (read_volatile(PWR_CR1) >> 9) & 0b11 != 0b01 {
            return false;
        }

        // 1. Raise the Flash latency first, and wait until it is applied.
        let acr = read_volatile(FLASH_ACR);
        write_volatile(
            FLASH_ACR,
            (acr & !FLASH_ACR_LATENCY_MASK) | FLASH_LATENCY_48MHZ,
        );
        while read_volatile(FLASH_ACR) & FLASH_ACR_LATENCY_MASK != FLASH_LATENCY_48MHZ {}

        // 2. MSIRANGE may only change while MSI is off or ready.
        while read_volatile(RCC_CR) & RCC_CR_MSIRDY == 0 {}
        let cr = read_volatile(RCC_CR);
        write_volatile(
            RCC_CR,
            (cr & !RCC_CR_MSIRANGE_MASK) | (MSI_RANGE_48MHZ << 4) | RCC_CR_MSIRGSEL,
        );
        while read_volatile(RCC_CR) & RCC_CR_MSIRDY == 0 {}
    }
    true
}

/// Core clock (HCLK1) in hertz, read from the RCC registers.
///
/// Returns `None` if the system clock is not MSI or the AHB prescaler is not 1, which this
/// firmware never configures.
pub fn hclk_hz() -> Option<u32> {
    // SAFETY: read-only accesses to fixed, aligned STM32WL RCC registers.
    let (cr, cfgr, csr) = unsafe {
        (
            read_volatile(RCC_CR),
            read_volatile(RCC_CFGR),
            read_volatile(RCC_CSR),
        )
    };
    let sws = (cfgr >> 2) & 0b11; // 00 = MSI
    let hpre = (cfgr >> 4) & 0xF; // 0000 = not divided
    if sws != 0 || hpre != 0 {
        return None;
    }
    let range = if cr & RCC_CR_MSIRGSEL != 0 {
        (cr >> 4) & 0xF
    } else {
        (csr >> 8) & 0xF // MSISRANGE, used after reset
    };
    MSI_RANGE_HZ.get(range as usize).copied()
}

/// Flash wait states currently applied.
pub fn flash_wait_states() -> u32 {
    // SAFETY: read-only access to the fixed, aligned FLASH_ACR register.
    unsafe { read_volatile(FLASH_ACR) & FLASH_ACR_LATENCY_MASK }
}

/// Converts a cycle count into microseconds at `clock_hz`.
pub fn cycles_to_us(cycles: f32, clock_hz: u32) -> f32 {
    cycles * 1_000_000.0 / clock_hz as f32
}

/// CPU-active energy estimate in microjoules for `cycles` at `clock_hz`, from the assumed
/// run current and supply voltage (not a measurement).
pub fn estimated_energy_uj(cycles: f32, clock_hz: u32) -> f32 {
    let power_uw = (ASSUMED_RUN_CURRENT_UA * ASSUMED_SUPPLY_MV / 1000) as f32;
    cycles_to_us(cycles, clock_hz) * power_uw / 1_000_000.0
}

/// Sets up the clock and prints the clock that is really applied. Returns HCLK in hertz.
///
/// Panics (halts, with `panic_halt`) if the clock cannot be read back: every printed time
/// would then be wrong.
pub fn init_and_report() -> u32 {
    let switched = init_clock_48mhz();
    let hz = hclk_hz().expect("system clock is not MSI without prescaler");
    rtt_target::rprintln!(
        "Clock: HCLK = {} MHz (read back from RCC), Flash wait states = {}{}",
        hz / 1_000_000,
        flash_wait_states(),
        if switched { "" } else { " (switch to 48 MHz refused: not in voltage range 1)" }
    );
    hz
}
