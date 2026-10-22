//! 📡 Industrial Asynchronous Event Ingestion (AER - Address-Event Representation).
//!
//! Real neuromorphic sensors (Prophesee/iniVation DVS cameras, silicon retinas, and cochleas)
//! do not emit dense periodic scalar frames, but rather asynchronous AER event streams.
//!
//! This module provides zero-allocation (`no_std`) data structures to ingest hardware
//! event packets (from DMA or interrupts) directly into [`SpikeBitset`].

use crate::domain::bitset::SpikeBitset;

/// Neuromorphic event in 1D AER format (e.g. cochlear audio channel index, tactile sensor).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AerSpike1d {
    /// Channel or source neuron address that fired the spike.
    pub address: u16,
    /// Relative timestamp in microseconds or ticks.
    pub timestamp: u16,
}

impl AerSpike1d {
    #[inline(always)]
    pub const fn new(address: u16, timestamp: u16) -> Self {
        Self { address, timestamp }
    }
}

/// 2D spatial vision neuromorphic event in AER format (DVS camera / neuromorphic vision sensor).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AerSpike2d {
    /// X coordinate (column).
    pub x: u16,
    /// Y coordinate (row).
    pub y: u16,
    /// Contrast change polarity (+1: brightness increase, 0: decrease).
    pub polarity: bool,
    /// Relative timestamp.
    pub timestamp: u32,
}

impl AerSpike2d {
    #[inline(always)]
    pub const fn new(x: u16, y: u16, polarity: bool, timestamp: u32) -> Self {
        Self {
            x,
            y,
            polarity,
            timestamp,
        }
    }

    /// Converts 2D spatial event into a 1D linear index (row-major).
    #[inline(always)]
    pub const fn to_linear_index(&self, width: usize) -> usize {
        (self.y as usize) * width + (self.x as usize)
    }

    /// Converts 2D spatial event with polarity (2 channels: ON/OFF) into a 1D linear index.
    #[inline(always)]
    pub const fn to_linear_index_with_polarity(&self, width: usize, height: usize) -> usize {
        let channel_offset = if self.polarity { 0 } else { width * height };
        channel_offset + (self.y as usize) * width + (self.x as usize)
    }
}

/// Streaming event-driven decoder to compact bitmask [`SpikeBitset`].
/// Zero dynamic allocation.
pub struct AerBitsetDecoder;

impl AerBitsetDecoder {
    /// Populates a bitmask [`SpikeBitset`] from a batch of 1D AER events.
    #[inline]
    pub fn decode_1d_batch<const WORDS: usize>(
        events: &[AerSpike1d],
        bitset: &mut SpikeBitset<WORDS>,
    ) {
        bitset.clear();
        for ev in events {
            bitset.set(ev.address as usize);
        }
    }

    /// Populates a bitmask [`SpikeBitset`] from a batch of 2D DVS spatial events (without polarity).
    #[inline]
    pub fn decode_2d_batch<const WORDS: usize>(
        events: &[AerSpike2d],
        width: usize,
        bitset: &mut SpikeBitset<WORDS>,
    ) {
        bitset.clear();
        for ev in events {
            let idx = ev.to_linear_index(width);
            bitset.set(idx);
        }
    }

    /// Populates a bitmask [`SpikeBitset`] from a 2D DVS batch with separated ON and OFF polarity channels.
    #[inline]
    pub fn decode_2d_bipolar_batch<const WORDS: usize>(
        events: &[AerSpike2d],
        width: usize,
        height: usize,
        bitset: &mut SpikeBitset<WORDS>,
    ) {
        bitset.clear();
        for ev in events {
            let idx = ev.to_linear_index_with_polarity(width, height);
            bitset.set(idx);
        }
    }
}

/// Zero-allocation DMA / interrupt circular buffer (SPSC lock-free) for asynchronous event streams.
/// Allows a hardware peripheral (DMA, SPI, GPIO ISR) to write AER events without stalling the CPU,
/// while the main inference loop consumes them in batches.
#[derive(Debug)]
pub struct AerDmaRingBuffer<T: Copy + Default, const CAPACITY: usize> {
    buffer: [T; CAPACITY],
    head: usize,
    tail: usize,
}

impl<T: Copy + Default, const CAPACITY: usize> Default for AerDmaRingBuffer<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const CAPACITY: usize> AerDmaRingBuffer<T, CAPACITY> {
    /// Creates a new empty circular buffer.
    pub fn new() -> Self {
        Self {
            buffer: [T::default(); CAPACITY],
            head: 0,
            tail: 0,
        }
    }

    /// Creates a new circular buffer from an initial element in a const context.
    pub const fn from_init(init: T) -> Self {
        Self {
            buffer: [init; CAPACITY],
            head: 0,
            tail: 0,
        }
    }

    /// Total capacity of the buffer.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    /// Number of elements currently available to read.
    #[inline]
    pub fn len(&self) -> usize {
        if self.head >= self.tail {
            self.head - self.tail
        } else {
            CAPACITY - (self.tail - self.head)
        }
    }

    /// Returns `true` if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    /// Returns `true` if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        let next = (self.head + 1) % CAPACITY;
        next == self.tail
    }

    /// Pushes an event from an interrupt or DMA transfer.
    /// Returns `Ok(())` on success, or `Err(item)` if buffer is full (prevents silent drops).
    #[inline]
    pub fn push(&mut self, item: T) -> Result<(), T> {
        let next = (self.head + 1) % CAPACITY;
        if next == self.tail {
            return Err(item);
        }
        self.buffer[self.head] = item;
        self.head = next;
        Ok(())
    }

    /// Extracts the oldest available event.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.head == self.tail {
            None
        } else {
            let item = self.buffer[self.tail];
            self.tail = (self.tail + 1) % CAPACITY;
            Some(item)
        }
    }

    /// Drains and decodes all accumulated 1D events directly into a [`SpikeBitset`].
    #[inline]
    pub fn drain_into_bitset_1d<const WORDS: usize>(&mut self, bitset: &mut SpikeBitset<WORDS>)
    where
        T: Into<AerSpike1d>,
    {
        bitset.clear();
        while let Some(item) = self.pop() {
            let ev: AerSpike1d = item.into();
            bitset.set(ev.address as usize);
        }
    }

    /// Resets the circular buffer pointers.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aer_1d_decoding() {
        let events = [
            AerSpike1d::new(0, 100),
            AerSpike1d::new(31, 105),
            AerSpike1d::new(32, 110),
        ];

        let mut bitset = SpikeBitset::<2>::new();
        AerBitsetDecoder::decode_1d_batch(&events, &mut bitset);

        assert!(bitset.get(0));
        assert!(bitset.get(31));
        assert!(bitset.get(32));
        assert!(!bitset.get(1));
        assert_eq!(bitset.count_ones(), 3);
    }

    #[test]
    fn test_aer_2d_spatial_decoding() {
        let width = 16;
        let height = 16;
        let events = [
            AerSpike2d::new(2, 3, true, 100),  // index = 3*16 + 2 = 50
            AerSpike2d::new(5, 1, false, 110), // channel off: 256 + 1*16 + 5 = 277
        ];

        let mut bitset = SpikeBitset::<16>::new();
        AerBitsetDecoder::decode_2d_bipolar_batch(&events, width, height, &mut bitset);

        assert!(bitset.get(50));
        assert!(bitset.get(277));
        assert!(!bitset.get(0));
        assert_eq!(bitset.count_ones(), 2);
    }

    #[test]
    fn test_aer_dma_ring_buffer_fifo_and_overflow() {
        let mut ring = AerDmaRingBuffer::<AerSpike1d, 4>::new();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);

        // Push 3 elements (max usable capacity for size 4 with sentinel = 3)
        assert!(ring.push(AerSpike1d::new(10, 100)).is_ok());
        assert!(ring.push(AerSpike1d::new(20, 101)).is_ok());
        assert!(ring.push(AerSpike1d::new(30, 102)).is_ok());
        assert_eq!(ring.len(), 3);
        assert!(ring.is_full());

        // Overflow detected
        let overflow_ev = AerSpike1d::new(40, 103);
        assert_eq!(ring.push(overflow_ev), Err(overflow_ev));

        // Drain into bitset
        let mut bitset = SpikeBitset::<2>::new();
        ring.drain_into_bitset_1d(&mut bitset);
        assert!(ring.is_empty());
        assert!(bitset.get(10));
        assert!(bitset.get(20));
        assert!(bitset.get(30));
        assert!(!bitset.get(40));
    }
}
