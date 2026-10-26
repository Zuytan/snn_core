//! C API bridge for snn_core.
//!
//! Exports unmangled C ABI symbols (`snn_*`) to be linked with any C/C++ toolchain.
//! The functions live in [`ffi`]; they are re-exported here so that integration tests and
//! Rust users can call them directly.
//!
//! The BPTN symbols (`snn_bptn_*`) are compiled in by the default `bptn` feature, matching
//! `include/snn_core.h`.
//!
//! On bare-metal targets (`target_os = "none"`) the crate is `no_std` and provides a minimal
//! panic handler that parks the core; with `panic = "abort"` (set for the release profile)
//! no unwinding machinery is needed. On hosted targets it links `std`.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "none"))]
extern crate std;

pub mod ffi;

pub use ffi::*;

#[cfg(all(target_os = "none", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
