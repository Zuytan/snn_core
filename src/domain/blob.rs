//! Versioned binary container for a model stored in Flash (`.snnb`).
//!
//! The container lets a model be shipped as data (OTA update, external Flash, file) instead of being
//! compiled into the firmware as Rust statics. [`ModelBlob::parse`] checks the whole file up front
//! (magic, version, length, CRC-32, structural invariants) and refuses anything it does not
//! understand; after that [`ModelBlob::load_layers`] hands out [`LayerDescriptor`]s that borrow the
//! tables **in place** (no copy, no allocation).
//!
//! # Format, version 1.0 (little-endian)
//!
//! ```text
//! offset  size  field
//! 0       4     magic            "SNNB"
//! 4       2     version_major    1
//! 6       2     version_minor    0
//! 8       4     flags            0 (any bit set => rejected: unknown must-understand feature)
//! 12      4     total_len        bytes in the whole blob, CRC trailer included
//! 16      4     num_layers
//! 20      4     header_len       32 in v1.0 (layer records start here)
//! 24      4     record_len       64 in v1.0
//! 28      4     reserved         0
//! header_len + i * record_len     layer record i (16 x u32, below)
//! ...           data tables, each starting on a 4-byte boundary
//! total_len-4   4     crc32      IEEE CRC-32 of bytes [0, total_len - 4)
//!
//! layer record (u32 words): kind, in_features, out_features, nnz,
//!   threshold, decay_factor, reset_potential, tonic_bias (i32),
//!   refractory_period, reset_mode (0 hard, 1 subtract), quant_scale (i32),
//!   bias_off (0 = none), row_ptrs_off, targets_off, weights_off, reserved (0)
//! kind 0: CSR, weights i32 Q16.16 [nnz]      kind 1: CSR, weights i8 [nnz] with shared quant_scale
//! row_ptrs: u16 [in_features + 1]   targets: u16 [nnz]   bias: i32 [out_features]
//! ```
//!
//! Compatibility rule: a reader accepts any `version_minor` of its own `version_major`; minor
//! versions may only append fields to the header and to records (readers step by `header_len` and
//! `record_len`) and add optional tables. A change that old readers must not ignore either sets a
//! `flags` bit or increments `version_major`.
//!
//! The blob must start at a 4-byte aligned address and the host must be little-endian; otherwise
//! parsing fails with [`BlobError::MisalignedBlob`] / [`BlobError::UnsupportedEndianness`].

use crate::domain::engine::LayerDescriptor;
use crate::domain::models::lif::{LifConfig, ResetMode};
use crate::domain::numeric::Fix32;
use crate::domain::rom_csr::{QuantizedRomCsrSynapses, RomCsrSynapses};

/// First four bytes of every blob.
pub const BLOB_MAGIC: [u8; 4] = *b"SNNB";
/// Major version understood by this reader.
pub const BLOB_VERSION_MAJOR: u16 = 1;
/// Minor version written by this crate.
pub const BLOB_VERSION_MINOR: u16 = 0;

const HEADER_LEN: usize = 32;
const RECORD_LEN: usize = 64;
const KIND_FIX32: u32 = 0;
const KIND_INT8: u32 = 1;

/// Why a blob was refused. Nothing is partially accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobError {
    /// Fewer bytes than the fixed header plus CRC.
    TooShort,
    /// The first four bytes are not `SNNB`.
    BadMagic,
    /// `version_major` differs from [`BLOB_VERSION_MAJOR`].
    UnsupportedVersion { major: u16, minor: u16 },
    /// A flag bit this reader does not know is set.
    UnsupportedFlags(u32),
    /// `total_len` does not match the number of bytes provided.
    LengthMismatch { declared: u32, actual: usize },
    /// The stored CRC-32 does not match the content.
    ChecksumMismatch { stored: u32, computed: u32 },
    /// The blob does not start at a 4-byte aligned address.
    MisalignedBlob,
    /// The host is big-endian; v1 tables are little-endian and read in place.
    UnsupportedEndianness,
    /// Header or record sizes are smaller than this version defines.
    BadLayout,
    /// A layer uses a `kind` this reader does not know.
    UnknownLayerKind(u32),
    /// A table offset is misaligned, overlaps the records or lies outside the blob.
    BadOffset { layer: u32 },
    /// A structural invariant is violated (monotonic `row_ptrs`, targets in range, chaining, ...).
    Invalid { layer: u32, reason: &'static str },
    /// The scratch array given to [`ModelBlob::load_layers`] is too small.
    TooManyLayers { needed: usize, capacity: usize },
}

impl BlobError {
    /// Stable numeric code (0 is success) used by the C API.
    pub const fn code(&self) -> i32 {
        match self {
            Self::TooShort => 1,
            Self::BadMagic => 2,
            Self::UnsupportedVersion { .. } => 3,
            Self::UnsupportedFlags(_) => 4,
            Self::LengthMismatch { .. } => 5,
            Self::ChecksumMismatch { .. } => 6,
            Self::MisalignedBlob => 7,
            Self::UnsupportedEndianness => 8,
            Self::BadLayout => 9,
            Self::UnknownLayerKind(_) => 10,
            Self::BadOffset { .. } => 11,
            Self::Invalid { .. } => 12,
            Self::TooManyLayers { .. } => 13,
        }
    }
}

/// IEEE 802.3 CRC-32 (reflected, polynomial 0xEDB88320), bitwise: no lookup table in Flash.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (!(crc & 1)).wrapping_add(1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// A validated model blob borrowing its bytes.
#[derive(Clone, Copy, Debug)]
pub struct ModelBlob<'a> {
    bytes: &'a [u8],
    num_layers: usize,
    header_len: usize,
    record_len: usize,
    minor: u16,
}

impl<'a> ModelBlob<'a> {
    /// Validates `bytes` completely and returns a handle on success.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, BlobError> {
        if !cfg!(target_endian = "little") {
            return Err(BlobError::UnsupportedEndianness);
        }
        if bytes.len() < HEADER_LEN + 4 {
            return Err(BlobError::TooShort);
        }
        if bytes[0..4] != BLOB_MAGIC {
            return Err(BlobError::BadMagic);
        }
        let major = rd_u16(bytes, 4);
        let minor = rd_u16(bytes, 6);
        if major != BLOB_VERSION_MAJOR {
            return Err(BlobError::UnsupportedVersion { major, minor });
        }
        let flags = rd_u32(bytes, 8);
        if flags != 0 {
            return Err(BlobError::UnsupportedFlags(flags));
        }
        let total_len = rd_u32(bytes, 12);
        if total_len as usize != bytes.len() {
            return Err(BlobError::LengthMismatch {
                declared: total_len,
                actual: bytes.len(),
            });
        }
        let body = &bytes[..bytes.len() - 4];
        let stored = rd_u32(bytes, bytes.len() - 4);
        let computed = crc32(body);
        if stored != computed {
            return Err(BlobError::ChecksumMismatch { stored, computed });
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            return Err(BlobError::MisalignedBlob);
        }
        let num_layers = rd_u32(bytes, 16) as usize;
        let header_len = rd_u32(bytes, 20) as usize;
        let record_len = rd_u32(bytes, 24) as usize;
        if header_len < HEADER_LEN || record_len < RECORD_LEN || !header_len.is_multiple_of(4) {
            return Err(BlobError::BadLayout);
        }
        if rd_u32(bytes, 28) != 0 {
            return Err(BlobError::BadLayout);
        }
        let records_end = num_layers
            .checked_mul(record_len)
            .and_then(|n| n.checked_add(header_len))
            .ok_or(BlobError::BadLayout)?;
        if num_layers == 0 || records_end > body.len() {
            return Err(BlobError::BadLayout);
        }
        let blob = Self {
            bytes,
            num_layers,
            header_len,
            record_len,
            minor,
        };
        blob.validate_layers(records_end)?;
        Ok(blob)
    }

    /// `(major, minor)` of the file.
    pub fn version(&self) -> (u16, u16) {
        (BLOB_VERSION_MAJOR, self.minor)
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn record(&self, i: usize) -> usize {
        self.header_len + i * self.record_len
    }

    /// Resolves a table: `count` elements of `elem` bytes at `off`, 4-byte aligned, after the
    /// records and before the CRC trailer.
    fn table(
        &self,
        layer: usize,
        off: u32,
        count: usize,
        elem: usize,
        records_end: usize,
    ) -> Result<(usize, usize), BlobError> {
        let bad = BlobError::BadOffset {
            layer: layer as u32,
        };
        let off = off as usize;
        let len = count.checked_mul(elem).ok_or(bad)?;
        let end = off.checked_add(len).ok_or(bad)?;
        if !off.is_multiple_of(4) || off < records_end || end > self.bytes.len() - 4 {
            return Err(bad);
        }
        Ok((off, len))
    }

    fn validate_layers(&self, records_end: usize) -> Result<(), BlobError> {
        let b = self.bytes;
        let mut prev_out: Option<usize> = None;
        for i in 0..self.num_layers {
            let r = self.record(i);
            let li = i as u32;
            let inv = |reason| BlobError::Invalid { layer: li, reason };
            let kind = rd_u32(b, r);
            if kind != KIND_FIX32 && kind != KIND_INT8 {
                return Err(BlobError::UnknownLayerKind(kind));
            }
            let in_f = rd_u32(b, r + 4) as usize;
            let out_f = rd_u32(b, r + 8) as usize;
            let nnz = rd_u32(b, r + 12) as usize;
            if in_f == 0 || out_f == 0 {
                return Err(inv("zero-sized layer"));
            }
            if in_f > 65_535 || out_f > 65_536 || nnz > 65_535 {
                return Err(inv("dimension exceeds u16 indexing"));
            }
            if rd_u32(b, r + 32) > u32::from(u16::MAX) {
                return Err(inv("refractory period exceeds u16"));
            }
            if rd_u32(b, r + 36) > 1 {
                return Err(inv("unknown reset mode"));
            }
            if rd_u32(b, r + 60) != 0 {
                return Err(inv("reserved field is not zero"));
            }
            if kind == KIND_FIX32 && rd_u32(b, r + 40) != 0 {
                return Err(inv("quant_scale set on a Fix32 layer"));
            }
            if let Some(p) = prev_out
                && p != in_f
            {
                return Err(inv("in_features differs from previous out_features"));
            }
            prev_out = Some(out_f);

            let (rp_off, _) = self.table(i, rd_u32(b, r + 48), in_f + 1, 2, records_end)?;
            let (tg_off, _) = self.table(i, rd_u32(b, r + 52), nnz, 2, records_end)?;
            let w_elem = if kind == KIND_FIX32 { 4 } else { 1 };
            self.table(i, rd_u32(b, r + 56), nnz, w_elem, records_end)?;
            let bias_off = rd_u32(b, r + 44);
            if bias_off != 0 {
                self.table(i, bias_off, out_f, 4, records_end)?;
            }

            // CSR invariants (docs/binary_format_spec.md section 3).
            if rd_u16(b, rp_off) != 0 {
                return Err(inv("row_ptrs[0] is not 0"));
            }
            let mut last = 0usize;
            for k in 0..=in_f {
                let v = rd_u16(b, rp_off + 2 * k) as usize;
                if v < last {
                    return Err(inv("row_ptrs not monotonic"));
                }
                last = v;
            }
            if last != nnz {
                return Err(inv("row_ptrs[in_features] != nnz"));
            }
            for k in 0..nnz {
                if rd_u16(b, tg_off + 2 * k) as usize >= out_f {
                    return Err(inv("target index out of range"));
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::cast_ptr_alignment)] // alignment checked in `parse`, see SAFETY
    fn u16_table(&self, off: usize, count: usize) -> &'a [u16] {
        // SAFETY: `validate_layers` checked that [off, off + 2*count) lies inside the blob and that
        // `off` is a multiple of 4; `parse` checked the blob base is 4-byte aligned, so the pointer is
        // 2-aligned. The bytes are little-endian and the host was checked little-endian, and every bit
        // pattern is a valid `u16`. The slice borrows `self.bytes` for `'a`, which is never mutated.
        unsafe { core::slice::from_raw_parts(self.bytes.as_ptr().add(off).cast::<u16>(), count) }
    }

    #[allow(clippy::cast_ptr_alignment)] // alignment checked in `parse`, see SAFETY
    fn fix32_table(&self, off: usize, count: usize) -> &'a [Fix32] {
        // SAFETY: as `u16_table`, with 4-byte alignment; `Fix32` is `repr(transparent)` over `i32`
        // and every bit pattern is valid.
        unsafe { core::slice::from_raw_parts(self.bytes.as_ptr().add(off).cast::<Fix32>(), count) }
    }

    fn i8_table(&self, off: usize, count: usize) -> &'a [i8] {
        // SAFETY: bounds checked in `validate_layers`; `i8` has size and alignment 1.
        unsafe { core::slice::from_raw_parts(self.bytes.as_ptr().add(off).cast::<i8>(), count) }
    }

    /// Borrowed view of layer `i` (`None` if out of range), for bindings that build their own descriptors.
    pub fn layer(&self, i: usize) -> Option<BlobLayerView<'a>> {
        if i >= self.num_layers {
            return None;
        }
        let b = self.bytes;
        let r = self.record(i);
        let kind = rd_u32(b, r);
        let in_features = rd_u32(b, r + 4) as usize;
        let out_features = rd_u32(b, r + 8) as usize;
        let nnz = rd_u32(b, r + 12) as usize;
        let lif = LifConfig::with_bias(
            Fix32(rd_u32(b, r + 16) as i32),
            Fix32(rd_u32(b, r + 20) as i32),
            Fix32(rd_u32(b, r + 24) as i32),
            rd_u32(b, r + 32) as u16,
            Fix32(rd_u32(b, r + 28) as i32),
        )
        .with_reset_mode(if rd_u32(b, r + 36) == 1 {
            ResetMode::Subtract
        } else {
            ResetMode::Hard
        });
        let w_off = rd_u32(b, r + 56) as usize;
        let bias_off = rd_u32(b, r + 44) as usize;
        let (weights_fix32, weights_i8) = if kind == KIND_FIX32 {
            (Some(self.fix32_table(w_off, nnz)), None)
        } else {
            (None, Some(self.i8_table(w_off, nnz)))
        };
        Some(BlobLayerView {
            in_features,
            out_features,
            lif,
            row_ptrs: self.u16_table(rd_u32(b, r + 48) as usize, in_features + 1),
            targets: self.u16_table(rd_u32(b, r + 52) as usize, nnz),
            weights_fix32,
            weights_i8,
            quant_scale: Fix32(rd_u32(b, r + 40) as i32),
            bias: (bias_off != 0).then(|| self.fix32_table(bias_off, out_features)),
        })
    }

    /// Fills `out` with one [`LayerDescriptor`] per layer, borrowing the tables in place, and returns
    /// the number of layers. Build the model with `ModelDescriptor::new(&out[..n])`.
    pub fn load_layers(&self, out: &mut [LayerDescriptor<'a>]) -> Result<usize, BlobError> {
        if out.len() < self.num_layers {
            return Err(BlobError::TooManyLayers {
                needed: self.num_layers,
                capacity: out.len(),
            });
        }
        for (i, slot) in out.iter_mut().enumerate().take(self.num_layers) {
            let v = self.layer(i).ok_or(BlobError::BadLayout)?;
            *slot = match (v.weights_fix32, v.weights_i8) {
                (Some(w), _) => LayerDescriptor::new(
                    v.in_features,
                    v.out_features,
                    RomCsrSynapses::new(v.row_ptrs, v.targets, w),
                    v.lif,
                    v.bias,
                ),
                (None, Some(w)) => LayerDescriptor::new_quantized_i8(
                    v.in_features,
                    v.out_features,
                    QuantizedRomCsrSynapses::new(v.row_ptrs, v.targets, w, v.quant_scale),
                    v.lif,
                    v.bias,
                ),
                (None, None) => return Err(BlobError::BadLayout),
            };
        }
        Ok(self.num_layers)
    }
}

/// One validated layer, borrowing the blob. Exactly one of the weight tables is `Some`.
#[derive(Clone, Copy, Debug)]
pub struct BlobLayerView<'a> {
    pub in_features: usize,
    pub out_features: usize,
    pub lif: LifConfig,
    pub row_ptrs: &'a [u16],
    pub targets: &'a [u16],
    pub weights_fix32: Option<&'a [Fix32]>,
    pub weights_i8: Option<&'a [i8]>,
    /// Shared scale of the INT8 weights (zero for Fix32 layers).
    pub quant_scale: Fix32,
    pub bias: Option<&'a [Fix32]>,
}

/// One layer to serialise with [`encode_model`].
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct BlobLayer<'a> {
    pub in_features: u32,
    pub out_features: u32,
    pub row_ptrs: &'a [u16],
    pub targets: &'a [u16],
    pub weights: BlobWeights<'a>,
    pub bias: Option<&'a [i32]>,
    pub lif: LifConfig,
}

/// Weight table of a [`BlobLayer`].
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub enum BlobWeights<'a> {
    /// Q16.16 raw values.
    Fix32(&'a [i32]),
    /// INT8 values and the shared Q16.16 scale.
    Int8 { weights: &'a [i8], scale: i32 },
}

/// Serialises a model into the v1.0 container (host tool and tests).
#[cfg(feature = "alloc")]
pub fn encode_model(layers: &[BlobLayer<'_>]) -> alloc::vec::Vec<u8> {
    use alloc::vec::Vec;

    fn put32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn pad4(v: &mut Vec<u8>) {
        while !v.len().is_multiple_of(4) {
            v.push(0);
        }
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&BLOB_MAGIC);
    out.extend_from_slice(&BLOB_VERSION_MAJOR.to_le_bytes());
    out.extend_from_slice(&BLOB_VERSION_MINOR.to_le_bytes());
    put32(&mut out, 0);
    put32(&mut out, 0); // total_len, patched below
    put32(&mut out, layers.len() as u32);
    put32(&mut out, HEADER_LEN as u32);
    put32(&mut out, RECORD_LEN as u32);
    put32(&mut out, 0);

    let records_at = out.len();
    out.resize(records_at + layers.len() * RECORD_LEN, 0);
    for (i, l) in layers.iter().enumerate() {
        pad4(&mut out);
        let rp_off = out.len();
        for v in l.row_ptrs {
            out.extend_from_slice(&v.to_le_bytes());
        }
        pad4(&mut out);
        let tg_off = out.len();
        for v in l.targets {
            out.extend_from_slice(&v.to_le_bytes());
        }
        pad4(&mut out);
        let w_off = out.len();
        let (kind, scale) = match &l.weights {
            BlobWeights::Fix32(w) => {
                for v in *w {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (KIND_FIX32, 0i32)
            }
            BlobWeights::Int8 { weights, scale } => {
                out.extend(weights.iter().map(|v| *v as u8));
                (KIND_INT8, *scale)
            }
        };
        pad4(&mut out);
        let bias_off = match l.bias {
            Some(b) => {
                let o = out.len();
                for v in b {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                o
            }
            None => 0,
        };
        let rec = [
            kind,
            l.in_features,
            l.out_features,
            l.targets.len() as u32,
            l.lif.threshold.0 as u32,
            l.lif.decay_factor.0 as u32,
            l.lif.reset_potential.0 as u32,
            l.lif.bias.0 as u32,
            u32::from(l.lif.refractory_period),
            u32::from(l.lif.reset_mode == ResetMode::Subtract),
            scale as u32,
            bias_off as u32,
            rp_off as u32,
            tg_off as u32,
            w_off as u32,
            0,
        ];
        for (k, v) in rec.iter().enumerate() {
            let at = records_at + i * RECORD_LEN + 4 * k;
            out[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    pad4(&mut out);
    let total = (out.len() + 4) as u32;
    out[12..16].copy_from_slice(&total.to_le_bytes());
    let crc = crc32(&out);
    put32(&mut out, crc);
    out
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::domain::engine::{AlignedBuffer, ModelDescriptor};
    use alloc::vec::Vec;

    /// 2-layer model: 2 -> 2 (Fix32) -> 2 (INT8 with bias).
    fn sample_blob() -> Vec<u8> {
        let lif = LifConfig::new(Fix32::ONE, Fix32::from_bits(58982), Fix32::ZERO, 1)
            .with_reset_mode(ResetMode::Subtract);
        encode_model(&[
            BlobLayer {
                in_features: 2,
                out_features: 2,
                row_ptrs: &[0, 1, 2],
                targets: &[0, 1],
                weights: BlobWeights::Fix32(&[98_304, 65_536]),
                bias: None,
                lif,
            },
            BlobLayer {
                in_features: 2,
                out_features: 2,
                row_ptrs: &[0, 1, 2],
                targets: &[1, 0],
                weights: BlobWeights::Int8 {
                    weights: &[100, -50],
                    scale: 1024,
                },
                bias: Some(&[0, 0]),
                lif,
            },
        ])
    }

    /// Copies `bytes` into a 4-byte aligned buffer (a `Vec<u8>` may not be aligned).
    fn aligned<const N: usize>(bytes: &[u8]) -> AlignedBuffer<N> {
        let mut buf = AlignedBuffer::<N>::new();
        buf.as_mut_bytes()[..bytes.len()].copy_from_slice(bytes);
        buf
    }

    fn refresh_crc(b: &mut [u8]) {
        let n = b.len() - 4;
        let c = crc32(&b[..n]);
        b[n..].copy_from_slice(&c.to_le_bytes());
    }

    macro_rules! with_blob {
        ($bytes:expr, $name:ident, $body:block) => {{
            let raw: Vec<u8> = $bytes;
            let mut buf = aligned::<1024>(&raw);
            let $name: &[u8] = &buf.as_mut_bytes()[..raw.len()];
            $body
        }};
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn round_trip_runs_inference() {
        with_blob!(sample_blob(), bytes, {
            let blob = ModelBlob::parse(bytes).expect("valid blob");
            assert_eq!(blob.version(), (1, 0));
            assert_eq!(blob.num_layers(), 2);
            let mut layers = [LayerDescriptor::EMPTY; 4];
            let n = blob.load_layers(&mut layers).expect("capacity");
            assert_eq!(n, 2);
            let model = ModelDescriptor::new(&layers[..n]);
            let mut sram = AlignedBuffer::<512>::new();
            assert!(model.sram_bytes() <= 512);
            let mut ctx = model.init_context(sram.as_mut_bytes()).expect("sram");
            let mut out = [0u32; 1];
            model
                .step(&mut ctx, &[Fix32::ONE, Fix32::ZERO], &mut out)
                .expect("step");
            // layer 0: input 0 (1.0 * 1.5) fires neuron 0; layer 1: neuron 0 -> target 1 with 100*scale.
            assert_eq!(out[0] & 0b11, 0b10);
        });
    }

    #[test]
    fn rejects_capacity_too_small() {
        with_blob!(sample_blob(), bytes, {
            let blob = ModelBlob::parse(bytes).unwrap();
            let mut layers = [LayerDescriptor::EMPTY; 1];
            assert_eq!(
                blob.load_layers(&mut layers),
                Err(BlobError::TooManyLayers {
                    needed: 2,
                    capacity: 1
                })
            );
        });
    }

    #[test]
    fn rejects_bad_magic_version_and_flags() {
        let mut v = sample_blob();
        v[0] = b'X';
        refresh_crc(&mut v);
        with_blob!(v, b, {
            assert_eq!(ModelBlob::parse(b).unwrap_err(), BlobError::BadMagic);
        });

        let mut v = sample_blob();
        v[4] = 2;
        refresh_crc(&mut v);
        with_blob!(v, b, {
            assert_eq!(
                ModelBlob::parse(b).unwrap_err(),
                BlobError::UnsupportedVersion { major: 2, minor: 0 }
            );
        });

        let mut v = sample_blob();
        v[8] = 1;
        refresh_crc(&mut v);
        with_blob!(v, b, {
            assert_eq!(
                ModelBlob::parse(b).unwrap_err(),
                BlobError::UnsupportedFlags(1)
            );
        });
    }

    #[test]
    fn accepts_newer_minor_version() {
        let mut v = sample_blob();
        v[6] = 7;
        refresh_crc(&mut v);
        with_blob!(v, b, {
            assert_eq!(ModelBlob::parse(b).unwrap().version(), (1, 7));
        });
    }

    #[test]
    fn rejects_truncation_and_wrong_length() {
        let v = sample_blob();
        for cut in [0, 3, 35, v.len() - 1] {
            with_blob!(v[..cut].to_vec(), b, {
                assert!(ModelBlob::parse(b).is_err(), "cut at {cut}");
            });
        }
        let mut longer = sample_blob();
        longer.extend_from_slice(&[0, 0, 0, 0]);
        with_blob!(longer, b, {
            assert!(matches!(
                ModelBlob::parse(b),
                Err(BlobError::LengthMismatch { .. })
            ));
        });
    }

    #[test]
    fn every_single_byte_corruption_is_rejected() {
        let v = sample_blob();
        for i in 0..v.len() {
            for flip in [0x01u8, 0x80, 0xFF] {
                let mut c = v.clone();
                c[i] ^= flip;
                with_blob!(c, b, {
                    assert!(
                        ModelBlob::parse(b).is_err(),
                        "byte {i} ^ {flip:#x} accepted"
                    );
                });
            }
        }
    }

    #[test]
    fn rejects_misaligned_blob() {
        let v = sample_blob();
        let mut buf = AlignedBuffer::<1024>::new();
        buf.as_mut_bytes()[1..=v.len()].copy_from_slice(&v);
        let shifted = &buf.as_mut_bytes()[1..=v.len()];
        assert_eq!(
            ModelBlob::parse(shifted).unwrap_err(),
            BlobError::MisalignedBlob
        );
    }

    /// Structural violations must be caught even when the CRC is valid (a buggy or hostile encoder).
    #[test]
    fn rejects_invalid_structure_with_valid_crc() {
        let v = sample_blob();
        let rec0 = HEADER_LEN;
        let rp0 = rd_u32(&v, rec0 + 48) as usize;
        let tg0 = rd_u32(&v, rec0 + 52) as usize;

        let cases: [(&str, Vec<u8>); 5] = {
            let mut non_mono = v.clone();
            non_mono[rp0 + 2] = 5;
            let mut bad_end = v.clone();
            bad_end[rp0 + 4] = 9;
            let mut bad_target = v.clone();
            bad_target[tg0] = 9;
            let mut bad_chain = v.clone();
            bad_chain[HEADER_LEN + RECORD_LEN + 4] = 3;
            let mut bad_off = v.clone();
            bad_off[rec0 + 56] = 0xFF;
            [
                ("monotonic", non_mono),
                ("row_ptrs[in]", bad_end),
                ("target", bad_target),
                ("chain", bad_chain),
                ("offset", bad_off),
            ]
        };
        for (name, mut c) in cases {
            refresh_crc(&mut c);
            with_blob!(c, b, {
                assert!(ModelBlob::parse(b).is_err(), "{name} accepted");
            });
        }
    }

    /// Mutations with a repaired CRC must never panic, and anything accepted must run safely.
    #[test]
    fn mutated_blobs_with_valid_crc_never_panic() {
        let base = sample_blob();
        let mut x: u32 = 0x1234_5678;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let mut accepted = 0;
        for _ in 0..20_000 {
            let mut c = base.clone();
            for _ in 0..=(next() % 3) {
                let at = (next() as usize) % (c.len() - 4);
                c[at] = next() as u8;
            }
            refresh_crc(&mut c);
            with_blob!(c, b, {
                if let Ok(blob) = ModelBlob::parse(b) {
                    accepted += 1;
                    let mut layers = [LayerDescriptor::EMPTY; 8];
                    let n = blob.load_layers(&mut layers).unwrap();
                    let model = ModelDescriptor::new(&layers[..n]);
                    let mut sram = AlignedBuffer::<4096>::new();
                    if let Ok(mut ctx) = model.init_context(sram.as_mut_bytes()) {
                        let inputs = [Fix32::ONE; 8];
                        let mut out = [0u32; 4];
                        let in_f = model.input_features().min(8);
                        let words = model.output_words().min(4);
                        let _ = model.step(&mut ctx, &inputs[..in_f], &mut out[..words]);
                    }
                }
            });
        }
        assert!(accepted > 0, "the mutator never produced a valid blob");
    }

    #[test]
    fn matches_the_golden_file_written_by_the_python_compiler() {
        #[repr(align(4))]
        struct Aligned<const N: usize>([u8; N]);
        const GOLDEN: &[u8] = include_bytes!("../../tests/data/model_v1.snnb");
        static ALIGNED: Aligned<{ GOLDEN.len() }> = {
            let mut a = [0u8; GOLDEN.len()];
            let mut i = 0;
            while i < GOLDEN.len() {
                a[i] = GOLDEN[i];
                i += 1;
            }
            Aligned(a)
        };
        let blob = ModelBlob::parse(&ALIGNED.0).expect("golden blob parses");
        assert_eq!(blob.version(), (1, 0));
        assert_eq!(blob.num_layers(), 2);
        // The Rust encoder must produce the same bytes as the Python one for the same model.
        assert_eq!(sample_blob(), GOLDEN);
    }
}
