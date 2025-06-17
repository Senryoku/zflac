const std = @import("std");
const builtin = @import("builtin");

const log = std.log.scoped(.zflac);
const log_frame = std.log.scoped(.zflac_frame);
const log_subframe = std.log.scoped(.zflac_subframe);
const log_residual = std.log.scoped(.zflac_residual);

const Signature: u32 = 0x664C6143;

channels: u8,
sample_rate: u24,
bits_per_sample: u8,
samples: []i16,

const MetadataHeader = packed struct {
    info: enum(u7) {
        Streaminfo = 0,
        Padding = 1,
        Application = 2,
        Seektable = 3,
        VorbisComment = 4,
        Cuesheet = 5,
        Picture = 6,
    },
    last_block: bool,
    length: u24,
};

const StreaminfoMetadata = struct {
    min_block_size: u16,
    max_block_size: u16,
    min_frame_size: u24,
    max_frame_size: u24,
    sample_rate: u20,
    channel_count: u3,
    sample_bit_depth: u5,
    number_of_samples: u36,
    md5: [16]u8,
};

const SampleRate = enum(u4) {
    StoredInMetadata = 0b0000, // Sample rate only stored in the streaminfo metadata block
    @"88.2kHz" = 0b0001, // 88.2 kHz
    @"176.4kHz" = 0b0010, // 176.4 kHz
    @"192kHz" = 0b0011, // 192 kHz
    @"8kHz" = 0b0100, // 8 kHz
    @"16kHz" = 0b0101, // 16 kHz
    @"22.05kHz" = 0b0110, // 22.05 kHz
    @"24kHz" = 0b0111, // 24 kHz
    @"32kHz" = 0b1000, // 32 kHz
    @"44.1kHz" = 0b1001, // 44.1 kHz
    @"48kHz" = 0b1010, // 48 kHz
    @"96kHz" = 0b1011, // 96 kHz
    Uncommon8b = 0b1100, // Uncommon sample rate in kHz, stored as an 8-bit number
    Uncommon16b = 0b1101, // Uncommon sample rate in Hz, stored as a 16-bit number
    Uncommon16bx10 = 0b1110, // Uncommon sample rate in Hz divided by 10, stored as a 16-bit number
    Forbidden = 0b1111, // Forbidden

    pub fn hz(self: @This()) u24 {
        return switch (self) {
            .@"88.2kHz" => 88200,
            .@"176.4kHz" => 176400,
            .@"192kHz" => 192000,
            .@"8kHz" => 8000,
            .@"16kHz" => 16000,
            .@"22.05kHz" => 22050,
            .@"24kHz" => 24000,
            .@"32kHz" => 32000,
            .@"44.1kHz" => 44100,
            .@"48kHz" => 48000,
            .@"96kHz" => 96000,
            .StoredInMetadata, .Uncommon8b, .Uncommon16b, .Uncommon16bx10, .Forbidden => unreachable,
        };
    }
};

const Channels = enum(u4) {
    Mono = 0b0000, // 1 channel: mono
    LR = 0b0001, // 2 channels: left, right
    LRC = 0b0010, // 3 channels: left, right, center
    @"4Channels" = 0b0011, // 4 channels: front left, front right, back left, back right
    @"5Channels" = 0b0100, // 5 channels: front left, front right, front center, back/surround left, back/surround right
    @"6Channels" = 0b0101, // 6 channels: front left, front right, front center, LFE, back/surround left, back/surround right
    @"7Channels" = 0b0110, // 7 channels: front left, front right, front center, LFE, back center, side left, side right
    @"8Channels" = 0b0111, // 8 channels: front left, front right, front center, LFE, back left, back right, side left, side right
    LRLeftSideStereo = 0b1000, // 2 channels: left, right; stored as left-side stereo
    LRSideRightStereo = 0b1001, // 2 channels: left, right; stored as side-right stereo
    LRMidSideStereo = 0b1010, // 2 channels: left, right; stored as mid-side stereo
    _, // Reserved

    pub fn count(self: @This()) u4 {
        return switch (self) {
            .Mono => 1,
            .LR => 2,
            .LRC => 3,
            .@"4Channels" => 4,
            .@"5Channels" => 5,
            .@"6Channels" => 6,
            .@"7Channels" => 7,
            .@"8Channels" => 8,
            .LRLeftSideStereo => 2,
            .LRSideRightStereo => 2,
            .LRMidSideStereo => 2,
            _ => 0,
        };
    }
};

const BitDepth = enum(u3) {
    StoredInMetadata = 0b000, // Bit depth only stored in the streaminfo metadata block
    @"8bps" = 0b001, // 8 bits per sample
    @"12bps" = 0b010, // 12 bits per sample
    Reserved = 0b011, // Reserved
    @"16bps" = 0b100, // 16 bits per sample
    @"20bps" = 0b101, // 20 bits per sample
    @"24bps" = 0b110, // 24 bits per sample
    @"32bps" = 0b111, // 32 bits per sample

    pub fn bps(self: @This()) u6 {
        return switch (self) {
            .@"8bps" => 8,
            .@"12bps" => 12,
            .@"16bps" => 16,
            .@"20bps" => 20,
            .@"24bps" => 24,
            .@"32bps" => 32,
            else => unreachable,
        };
    }
};

const FrameHeader = struct {
    frame_sync: u15,
    blocking_strategy: enum(u1) { Fixed = 0, Variable = 1 },
    block_size: u4,
    sample_rate: SampleRate,
    channels: Channels,
    bit_depth: BitDepth,
    zero: u1,
};

const SubframeHeader = packed struct {
    zero: u1,
    // 0b000000             Constant subframe
    // 0b000001             Verbatim subframe
    // 0b000010 - 0b000111  Reserved
    // 0b001000 - 0b001100  Subframe with a fixed predictor of order v-8; i.e., 0, 1, 2, 3 or 4
    // 0b001101 - 0b011111  Reserved
    // 0b100000 - 0b111111  Subframe with a linear predictor of order v-31; i.e., 1 through 32 (inclusive)
    subframe_type: u6,
    wasted_bit_flag: u1,
};

fn read_unary_integer(bit_reader: anytype) !u32 {
    var unary_integer: u32 = 0;
    while (try bit_reader.readBitsNoEof(u1, 1) == 0) unary_integer += 1;
    return unary_integer;
}

/// Reads a signed integer with a runtime known bit depth
fn read_signed_integer(comptime T: type, bit_reader: anytype, bit_depth: u6) !T {
    const container_type = switch (@bitSizeOf(T)) {
        8 => u8,
        16 => u16,
        32 => u32,
        64 => u64,
        else => @compileError("Unsupported container type: " ++ @typeName(T)),
    };
    var r = try bit_reader.readBitsNoEof(container_type, bit_depth);
    // Sign extend from bit_depth to container_type size
    if ((@as(container_type, 1) << @intCast(bit_depth - 1)) & r != 0) r |= @as(container_type, @truncate(0xFFFFFFFFFFFFFFFF)) << @intCast(bit_depth);
    return @bitCast(r);
}

inline fn read_unencoded_sample(bit_reader: anytype, wasted_bits: u6, bits_per_sample: u6) !i16 {
    if (wasted_bits > 0) {
        return @intCast(try read_signed_integer(i32, bit_reader, bits_per_sample - wasted_bits));
    } else {
        return @intCast(try read_signed_integer(i32, bit_reader, bits_per_sample));
    }
}

fn decode_residuals(allocator: std.mem.Allocator, block_size: u32, order: u32, bit_reader: anytype) ![]i32 {
    var residuals = try allocator.alloc(i32, block_size - order);
    errdefer allocator.free(residuals);

    const coding_method = try bit_reader.readBitsNoEof(u2, 2);
    if (coding_method >= 0b10) return error.InvalidResidualCodingMethod;
    const partition_order = try bit_reader.readBitsNoEof(u4, 4);

    log_residual.debug("    Residual decoding. coding_method: {d}, partition_order: {d}", .{ coding_method, partition_order });

    var partition_start_idx: u32 = 0;
    for (0..std.math.pow(u32, 2, partition_order)) |partition| {
        var count = (block_size >> partition_order);
        if (partition == 0) count -= order;
        const rice_parameter: u5 = try bit_reader.readBitsNoEof(u5, switch (coding_method) {
            0b00 => 4,
            0b01 => 5,
            else => unreachable,
        });
        log_residual.debug("      partition[{d}]: rice_parameter: {d}", .{ partition, rice_parameter });
        if ((coding_method == 0b00 and rice_parameter == 0b1111) or (coding_method == 0b01 and rice_parameter == 0b11111)) {
            // No rice parameter
            const bit_depth: u5 = try bit_reader.readBitsNoEof(u5, 5);
            for (0..count) |i|
                residuals[partition_start_idx + i] = try read_signed_integer(i32, bit_reader, bit_depth);
        } else {
            for (0..count) |i| {
                const quotient = try read_unary_integer(bit_reader);
                const remainder = try bit_reader.readBitsNoEof(u32, rice_parameter);
                const zigzag_encoded: u32 = (quotient << rice_parameter) + remainder;
                const residual: i32 = @bitCast((zigzag_encoded >> 1) ^ @as(u32, @bitCast(-@as(i32, @intCast(zigzag_encoded & 1)))));
                residuals[partition_start_idx + i] = residual;
            }
        }
        partition_start_idx += count;
    }

    return residuals;
}

pub fn decode(allocator: std.mem.Allocator, reader: anytype) !@This() {
    const signature = try reader.readInt(u32, .big);
    if (signature != Signature)
        return error.InvalidSignature;

    var first_audio_frame: usize = 0;

    var stream_info: ?StreaminfoMetadata = null;
    while (true) {
        var header = try reader.readStruct(MetadataHeader);
        header.length = @byteSwap(header.length);
        log.debug("header: {any}", .{header});

        switch (header.info) {
            .Streaminfo => {
                var bit_reader = std.io.bitReader(.big, reader);
                stream_info = .{
                    .min_block_size = try bit_reader.readBitsNoEof(u16, 16),
                    .max_block_size = try bit_reader.readBitsNoEof(u16, 16),
                    .min_frame_size = try bit_reader.readBitsNoEof(u24, 24),
                    .max_frame_size = try bit_reader.readBitsNoEof(u24, 24),
                    .sample_rate = try bit_reader.readBitsNoEof(u20, 20),
                    .channel_count = try bit_reader.readBitsNoEof(u3, 3),
                    .sample_bit_depth = try bit_reader.readBitsNoEof(u5, 5),
                    .number_of_samples = try bit_reader.readBitsNoEof(u36, 36),
                    .md5 = try reader.readBytesNoEof(16),
                };
                log.debug("  stream info: {any}", .{stream_info});
            },
            .Padding => {
                try reader.skipBytes(header.length, .{});
            },
            else => {
                log.warn(" Unhandled header: {s}", .{@tagName(header.info)});
                try reader.skipBytes(header.length, .{});
            },
        }

        if (header.last_block) {
            first_audio_frame = try reader.context.getPos() + header.length;
            break;
        }
    }

    if (stream_info == null)
        return error.MissingStreamInfo;

    var first_frame = true;
    var sample_rate: u24 = undefined;
    var channel_count: u4 = undefined;
    var bit_depth: BitDepth = undefined;
    var bits_per_sample: u6 = undefined;

    var samples = try allocator.alloc(i16, (@as(usize, stream_info.?.channel_count) + 1) * stream_info.?.number_of_samples);
    errdefer allocator.free(samples);

    //  try reader.context.seekTo(first_audio_frame);
    var frame_sample_offset: usize = 0;
    while (frame_sample_offset < samples.len) {
        var frame_header: FrameHeader = undefined;
        var bit_reader = std.io.bitReader(.big, reader);
        log_frame.debug("(reader offset: {d})", .{try reader.context.getPos()});
        frame_header.frame_sync = try bit_reader.readBitsNoEof(u15, 15);
        if (frame_header.frame_sync != (0xFFF8 >> 1))
            return error.InvalidFrameHeader;
        frame_header.blocking_strategy = @enumFromInt(try bit_reader.readBitsNoEof(u1, 1));
        frame_header.block_size = try bit_reader.readBitsNoEof(u4, 4);
        frame_header.sample_rate = @enumFromInt(try bit_reader.readBitsNoEof(u4, 4));
        frame_header.channels = @enumFromInt(try bit_reader.readBitsNoEof(u4, 4));
        frame_header.bit_depth = @enumFromInt(try bit_reader.readBitsNoEof(u3, 3));
        frame_header.zero = try bit_reader.readBitsNoEof(u1, 1);
        if (frame_header.zero != 0)
            return error.InvalidFrameHeader;
        log_frame.debug("frame header: {any} (frame_sample_offset: {d})", .{ frame_header, frame_sample_offset });

        const first_byte = try reader.readInt(u8, .big);
        if (first_byte == 0xFF) return error.InvalidFrameNumber;
        const byte_count = @clz(first_byte ^ 0xFF);
        var coded_number: u32 = (first_byte & (@as(u8, 0x7F) >> @intCast(byte_count)));
        if (byte_count > 0) {
            for (0..byte_count - 1) |_| {
                coded_number <<= 6;
                coded_number |= (try reader.readInt(u8, .big)) & 0x3F;
            }
        }
        switch (frame_header.blocking_strategy) {
            .Fixed => log_frame.debug("  Frame number: {d}", .{coded_number}),
            .Variable => log_frame.debug("  Sample number: {d}", .{coded_number}),
        }

        const block_size: u16 = switch (frame_header.block_size) {
            0b0000 => return error.InvalidFrameHeader, // Reserved
            0b0001 => 192,
            0b0010...0b0101 => |b| 144 * std.math.pow(u16, 2, b),
            // Uncommon block size
            0b0110 => @as(u16, try reader.readInt(u8, .big)) + 1,
            0b0111 => try reader.readInt(u16, .big) + 1,
            0b1000...0b1111 => |b| std.math.pow(u16, 2, b),
        };
        log_frame.debug("  block_size: {d}", .{block_size});

        const frame_sample_rate: u24 = switch (frame_header.sample_rate) {
            .StoredInMetadata => if (stream_info) |si| si.sample_rate else return error.InvalidFrameHeader,
            .Uncommon8b => try reader.readInt(u8, .big),
            .Uncommon16b => try reader.readInt(u16, .big),
            .Uncommon16bx10 => 10 * @as(u24, try reader.readInt(u16, .big)),
            .Forbidden => return error.InvalidFrameHeader,
            else => |sr| sr.hz(),
        };
        log_frame.debug("  frame_sample_rate: {d}", .{frame_sample_rate});

        if (first_frame) {
            sample_rate = frame_sample_rate;
            channel_count = frame_header.channels.count();
            bit_depth = frame_header.bit_depth;

            bits_per_sample = switch (frame_header.bit_depth) {
                .StoredInMetadata => if (stream_info) |si| @as(u6, si.sample_bit_depth) + 1 else return error.InvalidFrameHeader,
                else => |bd| bd.bps(),
            };

            first_frame = false;
        } else {
            // "Because not all environments in which FLAC decoders are used are able to cope with changes to these properties during playback, a decoder MAY choose to stop decoding on such a change."
            if (sample_rate != frame_sample_rate or channel_count != frame_header.channels.count() or bit_depth != frame_header.bit_depth) return error.InconsistentParameters;
        }

        const frame_header_crc = try reader.readInt(u8, .big);
        log_frame.debug("  frame_header_crc: {X:0>2}", .{frame_header_crc});

        // TODO: Check CRC
        // Finally, an 8-bit CRC follows the frame/sample number, an uncommon block size, or an uncommon sample rate (depending on whether the latter two are stored).
        // This CRC is initialized with 0 and has the polynomial x^8 + x^2 + x^1 + x^0. This CRC covers the whole frame header before the CRC, including the sync code.

        for (0..channel_count) |channel| {
            const subframe_header: SubframeHeader = .{
                .zero = try bit_reader.readBitsNoEof(u1, 1),
                .subframe_type = try bit_reader.readBitsNoEof(u6, 6),
                .wasted_bit_flag = try bit_reader.readBitsNoEof(u1, 1),
            };
            log_subframe.debug("  subframe_header[{d}]: {any} (first sample: {d})", .{ channel, subframe_header, frame_sample_offset + channel });
            if (subframe_header.zero != 0)
                return error.InvalidSubframeHeader;

            var wasted_bits: u6 = 0;
            if (subframe_header.wasted_bit_flag == 1) {
                wasted_bits = @intCast(try read_unary_integer(&bit_reader) + 1);
                log_subframe.debug("  wasted_bits: {d}", .{wasted_bits});
            }

            switch (subframe_header.subframe_type) {
                0b000000 => return error.Unimplemented, // Constant subframe
                0b000001 => { // Verbatim subframe
                    for (0..block_size) |i| {
                        samples[frame_sample_offset + channel_count * i + channel] = try read_unencoded_sample(&bit_reader, wasted_bits, bits_per_sample);
                        log_subframe.debug("    sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
                    }
                },
                0b001000...0b001100 => |t| { // Subframe with a fixed predictor of order v-8; i.e., 0, 1, 2, 3 or 4
                    const order: u3 = @intCast(t & 0b000111);
                    log_subframe.debug("  Subframe with a fixed predictor of order {d}", .{order});

                    const bits = switch (frame_header.channels) {
                        .LRLeftSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                        .LRSideRightStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                        .LRMidSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                        else => bits_per_sample,
                    };
                    for (0..order) |i| {
                        samples[frame_sample_offset + channel_count * i + channel] = try read_unencoded_sample(&bit_reader, wasted_bits, bits);
                        log_subframe.debug("    warmup_sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
                    }

                    const residuals = try decode_residuals(allocator, block_size, order, &bit_reader);
                    defer allocator.free(residuals);

                    for (0..block_size - order) |i| {
                        const idx = frame_sample_offset + channel_count * (order + i) + channel;
                        switch (order) {
                            0 => samples[idx] = @intCast(residuals[i]),
                            1 => samples[idx] = @intCast(residuals[i] + @as(i32, 1) * samples[idx - channel_count * 1]),
                            2 => samples[idx] = @intCast(residuals[i] + @as(i32, 2) * samples[idx - channel_count * 1] - @as(i32, 1) * samples[idx - channel_count * 2]),
                            3 => samples[idx] = @intCast(residuals[i] + @as(i32, 3) * samples[idx - channel_count * 1] - @as(i32, 3) * samples[idx - channel_count * 2] + @as(i32, 1) * samples[idx - channel_count * 3]),
                            4 => samples[idx] = @intCast(residuals[i] + @as(i32, 4) * samples[idx - channel_count * 1] - @as(i32, 6) * samples[idx - channel_count * 2] + @as(i32, 4) * samples[idx - channel_count * 3] - samples[idx - channel_count * 4]),
                            else => return error.InvalidSubframeHeader,
                        }
                    }
                },
                0b100000...0b111111 => |t| { // Subframe with a linear predictor of order v-31; i.e., 1 through 32 (inclusive)
                    const order: u6 = @intCast((t & 0b011111) + 1);
                    log_subframe.debug("  Subframe with a linear predictor of order: {d}", .{order});
                    // Unencoded warm-up samples (n = subframe's bits per sample * LPC order).
                    const bits = switch (frame_header.channels) {
                        .LRLeftSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                        .LRSideRightStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                        .LRMidSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                        else => bits_per_sample,
                    };
                    for (0..order) |i| {
                        samples[frame_sample_offset + channel_count * i + channel] = try read_unencoded_sample(&bit_reader, wasted_bits, bits);
                        log_subframe.debug("    warmup_sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
                    }
                    // (Predictor coefficient precision in bits)-1 (Note: 0b1111 is forbidden).
                    const coefficient_precision = (try bit_reader.readBitsNoEof(u4, 4)) + 1;
                    log_subframe.debug("    coefficient_precision: {d}", .{coefficient_precision});
                    // Prediction right shift needed in bits.
                    const coefficient_shift_right = try bit_reader.readBitsNoEof(u5, 5);
                    log_subframe.debug("    coefficient_shift_right: {d}", .{coefficient_shift_right});

                    // Predictor coefficients (n = predictor coefficient precision * LPC order).
                    var predictor_coefficient: [32]i16 = undefined;
                    for (0..order) |i| {
                        var r = try bit_reader.readBitsNoEof(u16, coefficient_precision);
                        if ((@as(u16, 1) << (coefficient_precision - 1)) & r != 0) r |= @as(u16, 0xFFFF) << coefficient_precision;
                        predictor_coefficient[i] = @bitCast(r);
                        log_subframe.debug("    predictor_coefficient[{d}]: {d}", .{ i, predictor_coefficient[i] });
                    }

                    const residuals = try decode_residuals(allocator, block_size, order, &bit_reader);
                    defer allocator.free(residuals);

                    for (0..block_size - order) |i| {
                        const idx = frame_sample_offset + channel_count * (order + i) + channel;
                        var predicted_without_shift: i32 = 0;
                        for (0..order) |o| {
                            predicted_without_shift += @as(i32, @intCast(samples[idx - channel_count * (1 + o)])) * predictor_coefficient[o];
                        }
                        const predicted = predicted_without_shift >> coefficient_shift_right;
                        samples[idx] = @intCast(predicted + residuals[i]);
                    }
                },
                0b000010...0b000111, 0b001101...0b011111 => return error.InvalidSubframeHeader, // Reserved
            }

            if (wasted_bits > 0) {
                for (0..block_size) |i| {
                    const idx = frame_sample_offset + channel_count * i + channel;
                    samples[idx] <<= @intCast(wasted_bits);
                }
            }
        }
        // NOTE: Last subframe is padded with zero bits to be byte aligned.
        bit_reader.alignToByte();

        const frame_crc = try reader.readInt(u16, .big);
        _ = frame_crc;
        // TODO: Check CRC
        // "Following this is a 16-bit CRC, initialized with 0, with the polynomial x^16 + x^15 + x^2 + x^0. This CRC covers the whole frame, excluding the 16-bit CRC but including the sync code."

        switch (frame_header.channels) {
            .LRLeftSideStereo => {
                for (0..block_size) |i| {
                    const idx = frame_sample_offset + channel_count * i;
                    samples[idx + 1] = samples[idx] - samples[idx + 1];
                }
            },
            .LRSideRightStereo => {
                for (0..block_size) |i| {
                    const idx = frame_sample_offset + channel_count * i;
                    samples[idx] += samples[idx + 1];
                }
            },
            .LRMidSideStereo => {
                for (0..block_size) |i| {
                    const idx = frame_sample_offset + channel_count * i;
                    var mid = @as(i32, samples[idx]) << 1;
                    const side = samples[idx + 1];
                    mid += side & 1;
                    samples[idx] = @intCast((mid + side) >> 1);
                    samples[idx + 1] = @intCast((mid - side) >> 1);
                }
            },
            else => {},
        }

        frame_sample_offset += channel_count * block_size;
    }

    var computed_md5: [16]u8 = undefined;
    switch (bit_depth) {
        .@"8bps" => {
            var d = std.crypto.hash.Md5.init(.{});
            for (0..samples.len) |i| {
                d.update(&[1]u8{@bitCast(@as(i8, @truncate(samples[i])))});
            }
            d.final(&computed_md5);
        },
        .@"12bps", .@"16bps" => std.crypto.hash.Md5.hash(std.mem.sliceAsBytes(samples), &computed_md5, .{}),
        else => return error.Unimplemented,
    }

    log.debug("samples: {d}", .{samples.len});

    if (!std.mem.eql(u8, &computed_md5, &stream_info.?.md5)) {
        if (builtin.is_test) {
            log.err("Invalid checksum", .{});
        } else {
            return error.InvalidChecksum;
        }
    }

    return .{
        .channels = channel_count,
        .sample_rate = sample_rate,
        .bits_per_sample = bits_per_sample,
        .samples = samples,
    };
}

pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.samples);
}
