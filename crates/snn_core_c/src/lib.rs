//! 🌐 C API Bridge for snn_core
//!
//! Exports unmangled C ABI symbols to be linked with any C/C++ compiler.

// Force inclusion of symbols from snn_core::ffi
pub use snn_core::ffi::*;
