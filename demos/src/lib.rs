//! Application-level helpers used by the examples and benchmarks of `snn_core`.
//!
//! These modules are deliberately outside the runtime crate: a CAN frame parser and the NARMA-10
//! benchmark system are not needed to run a spiking network on a microcontroller.

#![no_std]

pub mod can_parser;
pub mod narma;
