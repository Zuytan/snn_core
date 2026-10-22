//! 🚗 Universal CAN Frame Parser for Embedded Cybersecurity (`no_std`)
//!
//! Directly ingests real-world CAN bus traces without dynamic allocation on the critical path:
//! - **SocketCAN / Linux `candump` compact format**: `(1478198376.389427) can0 123#DEADBEEF`
//! - **SocketCAN / Linux `candump` standard format**: `can0  123   [8]  11 22 33 44 55 66 77 88`
//! - **HCRL Car-Hacking Dataset format**: `Timestamp,CAN_ID,DLC,D0,D1,D2,D3,D4,D5,D6,D7,Flag`
//! - **Generic CSV format**: `timestamp,can_id,payload_hex[,label]`

/// Decoded and normalized CAN frame with microsecond resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParsedCanFrame {
    /// Absolute timestamp in microseconds (µs).
    pub timestamp_us: u64,
    /// CAN arbitration identifier (11-bit standard or 29-bit extended).
    pub can_id: u32,
    /// Data Length Code (0 to 8 bytes).
    pub dlc: u8,
    /// Raw payload bytes (0 to 8 bytes).
    pub payload: [u8; 8],
    /// Ground-truth label: `Some(true)` if confirmed attack, `Some(false)` if healthy traffic, `None` if unknown.
    pub is_attack: Option<bool>,
}

impl ParsedCanFrame {
    /// Extracts a 16-bit scalar payload (Big-Endian word of the first two bytes).
    #[inline(always)]
    pub const fn payload_u16_be(&self) -> u16 {
        if self.dlc >= 2 {
            ((self.payload[0] as u16) << 8) | (self.payload[1] as u16)
        } else if self.dlc == 1 {
            self.payload[0] as u16
        } else {
            0
        }
    }

    /// Extracts a 16-bit scalar payload in Little-Endian.
    #[inline(always)]
    pub const fn payload_u16_le(&self) -> u16 {
        if self.dlc >= 2 {
            ((self.payload[1] as u16) << 8) | (self.payload[0] as u16)
        } else if self.dlc == 1 {
            self.payload[0] as u16
        } else {
            0
        }
    }
}

/// Converts an ASCII hex character ('0'..'9', 'a'..'f', 'A'..'F') to numerical value (0..15).
#[inline(always)]
const fn hex_char_to_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Decodes a hex string slice to a `u8` byte.
#[inline(always)]
fn parse_hex_u8(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() == 2 {
        let hi = hex_char_to_val(bytes[0])?;
        let lo = hex_char_to_val(bytes[1])?;
        Some((hi << 4) | lo)
    } else if bytes.len() == 1 {
        hex_char_to_val(bytes[0])
    } else {
        None
    }
}

/// Decodes a hex string slice to a 32-bit unsigned integer `u32`.
fn parse_hex_u32(s: &str) -> Option<u32> {
    let mut val = 0u32;
    for &b in s.as_bytes() {
        let digit = hex_char_to_val(b)? as u32;
        val = (val << 4) | digit;
    }
    Some(val)
}

/// Converts a floating-point timestamp string (e.g. `"1478198376.389427"` or `"1580000000.123456"`) to microseconds `u64`.
pub fn parse_timestamp_us(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.strip_prefix('(').unwrap_or(s);
    let s = s.strip_suffix(')').unwrap_or(s);

    if let Some((sec_str, frac_str)) = s.split_once('.') {
        let sec: u64 = sec_str.parse().ok()?;
        // Pad or truncate to 6 digits for microseconds
        let frac_bytes = frac_str.as_bytes();
        let mut frac_val: u64 = 0;
        let mut factor: u64 = 100_000;

        for i in 0..6 {
            if i < frac_bytes.len() {
                let digit = (frac_bytes[i] as char).to_digit(10)? as u64;
                frac_val += digit * factor;
            }
            factor /= 10;
        }

        Some(sec.saturating_mul(1_000_000).saturating_add(frac_val))
    } else {
        // Pure integer (already in seconds or microseconds)
        let val: u64 = s.parse().ok()?;
        if val > 1_000_000_000_000 {
            Some(val) // Already in microseconds
        } else {
            Some(val.saturating_mul(1_000_000))
        }
    }
}

/// Decodes a line in standard **HCRL Car-Hacking Dataset** format:
/// `Timestamp,CAN_ID,DLC,DATA[0],DATA[1],DATA[2],DATA[3],DATA[4],DATA[5],DATA[6],DATA[7],Flag`
/// Example: `1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R`
pub fn parse_hcrl_csv_line(line: &str) -> Option<ParsedCanFrame> {
    let mut parts = line.split(',');

    let ts_str = parts.next()?;
    let id_str = parts.next()?;
    let dlc_str = parts.next()?;

    let timestamp_us = parse_timestamp_us(ts_str)?;
    let can_id = parse_hex_u32(id_str.trim())?;
    let dlc: u8 = dlc_str.trim().parse().ok()?;

    let mut payload = [0u8; 8];
    let dlc_clamped = (dlc as usize).min(8);

    for p in payload.iter_mut().take(dlc_clamped) {
        if let Some(byte_str) = parts.next() {
            *p = parse_hex_u8(byte_str.trim())?;
        } else {
            break;
        }
    }

    // Label flag is at the end: 'R' (Regular / Normal) or 'T' (Tampered / Attack)
    let mut flag_str = None;
    for part in parts {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            flag_str = Some(trimmed);
        }
    }

    let is_attack = match flag_str {
        Some("T" | "Attack" | "1" | "True") => Some(true),
        Some("R" | "Normal" | "0" | "False") => Some(false),
        _ => None,
    };

    Some(ParsedCanFrame {
        timestamp_us,
        can_id,
        dlc,
        payload,
        is_attack,
    })
}

/// Decodes a line in compact **Linux SocketCAN `candump`** format:
/// `(1478198376.389427) can0 123#DEADBEEF` or `(1478198376.389427) can0 123#`
pub fn parse_candump_compact_line(line: &str) -> Option<ParsedCanFrame> {
    let mut parts = line.split_whitespace();

    let ts_str = parts.next()?;
    let _iface = parts.next()?; // e.g. "can0" or "vcan0"
    let packet_str = parts.next()?; // e.g. "123#0102030405060708"

    let timestamp_us = parse_timestamp_us(ts_str)?;
    let (id_str, data_str) = packet_str.split_once('#')?;

    let can_id = parse_hex_u32(id_str.trim())?;
    let data_str = data_str.trim();

    let mut payload = [0u8; 8];
    let data_len = data_str.len() / 2;
    let dlc = data_len.min(8) as u8;

    let bytes = data_str.as_bytes();
    for i in 0..(dlc as usize) {
        let hi = hex_char_to_val(bytes[i * 2])?;
        let lo = hex_char_to_val(bytes[i * 2 + 1])?;
        payload[i] = (hi << 4) | lo;
    }

    Some(ParsedCanFrame {
        timestamp_us,
        can_id,
        dlc,
        payload,
        is_attack: None,
    })
}

/// Decodes a line in standard extended **Linux SocketCAN `candump`** format:
/// `can0  123   [8]  11 22 33 44 55 66 77 88`
pub fn parse_candump_standard_line(line: &str) -> Option<ParsedCanFrame> {
    let mut parts = line.split_whitespace();

    let first = parts.next()?;
    let (timestamp_us, can_id_str) = if first.starts_with('(') {
        let ts = parse_timestamp_us(first)?;
        let _iface = parts.next()?;
        let id = parts.next()?;
        (ts, id)
    } else {
        // No initial timestamp: use ID directly
        let id = parts.next()?;
        (0u64, id)
    };

    let can_id = parse_hex_u32(can_id_str.trim())?;
    let dlc_part = parts.next()?; // "[8]"
    let dlc_str = dlc_part.trim_start_matches('[').trim_end_matches(']');
    let dlc: u8 = dlc_str.parse().ok()?;

    let mut payload = [0u8; 8];
    for p in payload.iter_mut().take((dlc as usize).min(8)) {
        if let Some(b_str) = parts.next() {
            *p = parse_hex_u8(b_str)?;
        }
    }

    Some(ParsedCanFrame {
        timestamp_us,
        can_id,
        dlc,
        payload,
        is_attack: None,
    })
}

/// Automatically decodes any CAN log line according to detected format.
pub fn parse_any_can_line(line: &str) -> Option<ParsedCanFrame> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
        return None;
    }

    if trimmed.contains(',') {
        parse_hcrl_csv_line(trimmed)
    } else if trimmed.contains('#') {
        parse_candump_compact_line(trimmed)
    } else if trimmed.contains('[') && trimmed.contains(']') {
        parse_candump_standard_line(trimmed)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hcrl_csv_regular_and_attack() {
        let regular_line = "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R";
        let frame = parse_hcrl_csv_line(regular_line).expect("Should parse regular HCRL line");
        assert_eq!(frame.timestamp_us, 1478198376389427);
        assert_eq!(frame.can_id, 0x0316);
        assert_eq!(frame.dlc, 8);
        assert_eq!(
            frame.payload,
            [0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6f]
        );
        assert_eq!(frame.is_attack, Some(false));
        assert_eq!(frame.payload_u16_be(), 0x0521);

        let attack_line = "1478198376.389500,0000,8,ff,ff,ff,ff,ff,ff,ff,ff,T";
        let frame_atk =
            parse_hcrl_csv_line(attack_line).expect("Should parse attack HCRL line");
        assert_eq!(frame_atk.can_id, 0x0000);
        assert_eq!(frame_atk.is_attack, Some(true));
        assert_eq!(frame_atk.payload[0], 0xff);
    }

    #[test]
    fn test_parse_candump_compact() {
        let line = "(1580000000.123456) can0 123#0102030405060708";
        let frame = parse_candump_compact_line(line).expect("Should parse compact candump");
        assert_eq!(frame.timestamp_us, 1580000000123456);
        assert_eq!(frame.can_id, 0x123);
        assert_eq!(frame.dlc, 8);
        assert_eq!(frame.payload, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_parse_candump_standard() {
        let line = "(1580000000.500000) can0  4A0   [8]  AA BB CC DD EE FF 00 11";
        let frame = parse_candump_standard_line(line).expect("Should parse standard candump");
        assert_eq!(frame.timestamp_us, 1580000000500000);
        assert_eq!(frame.can_id, 0x4A0);
        assert_eq!(frame.dlc, 8);
        assert_eq!(frame.payload[0], 0xAA);
        assert_eq!(frame.payload[1], 0xBB);
    }

    #[test]
    fn test_parse_any_can_line_dispatch() {
        let l1 = "1478198376.100000,0100,2,12,34,R";
        let f1 = parse_any_can_line(l1).unwrap();
        assert_eq!(f1.can_id, 0x0100);
        assert_eq!(f1.dlc, 2);

        let l2 = "(1478198376.200000) can0 200#DEAD";
        let f2 = parse_any_can_line(l2).unwrap();
        assert_eq!(f2.can_id, 0x200);
        assert_eq!(f2.dlc, 2);
    }
}
