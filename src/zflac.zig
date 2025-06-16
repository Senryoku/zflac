const std = @import("std");
const log = std.log.scoped(.zflac);

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
    blocking_strategy: enum(u1) { fixed = 0, variable = 1 },
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

fn decode_residuals(allocator: std.mem.Allocator, block_size: u32, order: u32, bit_reader: anytype) ![]i32 {
    var residuals = try allocator.alloc(i32, block_size - order);
    errdefer allocator.free(residuals);

    const coding_method = try bit_reader.readBitsNoEof(u2, 2);
    if (coding_method >= 0b10) return error.InvalidResidualCodingMethod;
    const partition_order = try bit_reader.readBitsNoEof(u4, 4);

    log.debug("    Residual decoding. coding_method: {d}, partition_order: {d}", .{ coding_method, partition_order });

    var partition_start_idx: u32 = 0;
    for (0..std.math.pow(u32, 2, partition_order)) |partition| {
        var count = (block_size >> partition_order);
        if (partition == 0) count -= order;
        const rice_parameter: u5 = try bit_reader.readBitsNoEof(u5, switch (coding_method) {
            0b00 => 4,
            0b01 => 5,
            else => unreachable,
        });
        log.debug("      partition[{d}]: rice_parameter: {d}", .{ partition, rice_parameter });
        if ((coding_method == 0b00 and rice_parameter == 0b1111) or (coding_method == 0b01 and rice_parameter == 0b11111)) {
            // No rice parameter
            const bit_depth: u5 = try bit_reader.readBitsNoEof(u5, 5);
            for (0..count) |i| {
                var r = try bit_reader.readBitsNoEof(u32, bit_depth);
                // Sign extend from bit_depth to 32b
                if ((@as(u32, 1) << (bit_depth - 1)) & r != 0) r |= @as(u32, 0xFFFFFFFF) << bit_depth;
                residuals[partition_start_idx + i] = @bitCast(r);
            }
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
                std.debug.print("  stream info: {any}\n", .{stream_info});
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
    var sample_rate: SampleRate = undefined;
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
        std.debug.print("frame header: {any} (frame_sample_offset: {d})\n", .{ frame_header, frame_sample_offset });

        if (first_frame) {
            sample_rate = frame_header.sample_rate;
            channel_count = frame_header.channels.count();
            bit_depth = frame_header.bit_depth;

            bits_per_sample = switch (frame_header.bit_depth) {
                .StoredInMetadata => if (stream_info) |si| @as(u6, si.sample_bit_depth) + 1 else return error.InvalidFrameHeader,
                else => |bd| bd.bps(),
            };

            first_frame = false;
        } else {
            // "Because not all environments in which FLAC decoders are used are able to cope with changes to these properties during playback, a decoder MAY choose to stop decoding on such a change."
            if (sample_rate != frame_header.sample_rate or channel_count != frame_header.channels.count() or bit_depth != frame_header.bit_depth) return error.InconsistentParameters;
        }

        switch (frame_header.blocking_strategy) {
            .fixed => {
                const frame_number = try reader.readInt(u8, .big);
                std.debug.print("  frame_number: {d}\n", .{frame_number});
            },
            .variable => return error.NotImplemented,
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
        log.debug("  block_size: {d}", .{block_size});

        const frame_header_crc = try reader.readInt(u8, .big);
        log.debug("  frame_header_crc: {X:0>2}", .{frame_header_crc});

        // TODO: Check CRC
        // Finally, an 8-bit CRC follows the frame/sample number, an uncommon block size, or an uncommon sample rate (depending on whether the latter two are stored).
        // This CRC is initialized with 0 and has the polynomial x^8 + x^2 + x^1 + x^0. This CRC covers the whole frame header before the CRC, including the sync code.

        for (0..channel_count) |channel| {
            const subframe_header: SubframeHeader = .{
                .zero = try bit_reader.readBitsNoEof(u1, 1),
                .subframe_type = try bit_reader.readBitsNoEof(u6, 6),
                .wasted_bit_flag = try bit_reader.readBitsNoEof(u1, 1),
            };
            log.debug("  subframe_header[{d}]: {any} (first sample: {d})", .{ channel, subframe_header, frame_sample_offset + channel });
            if (subframe_header.zero != 0)
                return error.InvalidSubframeHeader;

            var wasted_bits: u6 = 0;
            if (subframe_header.wasted_bit_flag == 1) {
                wasted_bits = @intCast(try read_unary_integer(&bit_reader) + 1);
                log.debug("  wasted_bits: {d}", .{wasted_bits});
            }

            switch (subframe_header.subframe_type) {
                0b000000 => return error.Unimplemented, // Constant subframe
                0b000001 => { // Verbatim subframe
                    for (0..block_size) |i| {
                        const idx = frame_sample_offset + frame_header.channels.count() * i + channel;
                        if (wasted_bits > 0) {
                            samples[idx] = try bit_reader.readBitsNoEof(i16, bits_per_sample - wasted_bits);
                            samples[idx] <<= @intCast(wasted_bits);
                        } else {
                            samples[idx] = try bit_reader.readBitsNoEof(i16, bits_per_sample);
                        }
                        log.debug("    sample: {d}", .{samples[idx]});
                    }
                },
                0b001000...0b001100 => |t| { // Subframe with a fixed predictor of order v-8; i.e., 0, 1, 2, 3 or 4
                    const order: u3 = @intCast(t & 0b000111);
                    log.debug("    order: {d}", .{order});

                    if (wasted_bits > 0) return error.Unimplemented;
                    for (0..order) |i| {
                        // FIXME: Wrong type
                        var warmup_sample = try bit_reader.readBitsNoEof(u32, switch (frame_header.channels) {
                            .LRLeftSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                            .LRSideRightStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                            .LRMidSideStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                            else => bits_per_sample,
                        });
                        if (bits_per_sample < 16) {
                            if ((@as(u16, 1) << @intCast(bits_per_sample - 1)) & warmup_sample != 0) warmup_sample |= @as(u16, 0xFFFF) << @intCast(bits_per_sample);
                        }
                        samples[frame_sample_offset + channel_count * i + channel] = @bitCast(@as(u16, @truncate(warmup_sample)));
                        log.debug("    warmup_sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
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
                    log.debug("    order: {d}", .{order});
                    // Unencoded warm-up samples (n = subframe's bits per sample * LPC order).
                    if (wasted_bits > 0) return error.Unimplemented;
                    const bits = switch (frame_header.channels) {
                        .LRLeftSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                        .LRSideRightStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                        .LRMidSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                        else => bits_per_sample,
                    };
                    for (0..order) |i| {
                        var warmup_sample = try bit_reader.readBitsNoEof(u32, bits);
                        if (bits < 16) {
                            if ((@as(u16, 1) << @intCast(bits - 1)) & warmup_sample != 0) warmup_sample |= @as(u16, 0xFFFF) << @intCast(bits);
                        }
                        samples[frame_sample_offset + frame_header.channels.count() * i + channel] = @bitCast(@as(u16, @truncate(warmup_sample)));
                        log.debug("    warmup_sample: {d}", .{samples[frame_sample_offset + frame_header.channels.count() * i + channel]});
                    }
                    // (Predictor coefficient precision in bits)-1 (Note: 0b1111 is forbidden).
                    const coefficient_precision = (try bit_reader.readBitsNoEof(u4, 4)) + 1;
                    log.debug("    coefficient_precision: {d}", .{coefficient_precision});
                    // Prediction right shift needed in bits.
                    const coefficient_shift_right = try bit_reader.readBitsNoEof(u5, 5);
                    log.debug("    coefficient_shift_right: {d}", .{coefficient_shift_right});

                    // Predictor coefficients (n = predictor coefficient precision * LPC order).
                    var predictor_coefficient: [32]i16 = undefined;
                    for (0..order) |i| {
                        var r = try bit_reader.readBitsNoEof(u16, coefficient_precision);
                        if ((@as(u16, 1) << (coefficient_precision - 1)) & r != 0) r |= @as(u16, 0xFFFF) << coefficient_precision;
                        predictor_coefficient[i] = @bitCast(r);
                        log.debug("    predictor_coefficient[{d}]: {d}", .{ i, predictor_coefficient[i] });
                    }

                    const residuals = try decode_residuals(allocator, block_size, order, &bit_reader);
                    defer allocator.free(residuals);

                    for (0..block_size - order) |i| {
                        const idx = frame_sample_offset + frame_header.channels.count() * (order + i) + channel;
                        var predicted_without_shift: i32 = 0;
                        for (0..order) |o| {
                            predicted_without_shift += @as(i32, @intCast(samples[idx - frame_header.channels.count() * (1 + o)])) * predictor_coefficient[o];
                        }
                        const predicted = predicted_without_shift >> coefficient_shift_right;
                        samples[idx] = @intCast(predicted + residuals[i]);
                    }
                },
                0b000010...0b000111, 0b001101...0b011111 => return error.InvalidSubframeHeader, // Reserved
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
                    const idx = frame_sample_offset + frame_header.channels.count() * i;
                    samples[idx + 1] = samples[idx] - samples[idx + 1];
                }
            },
            .LRSideRightStereo => {
                for (0..block_size) |i| {
                    const idx = frame_sample_offset + frame_header.channels.count() * i;
                    samples[idx] += samples[idx + 1];
                }
            },
            .LRMidSideStereo => {
                for (0..block_size) |i| {
                    const idx = frame_sample_offset + frame_header.channels.count() * i;
                    var mid = @as(i32, samples[idx]) << 1;
                    const side = samples[idx + 1];
                    mid += side & 1;
                    samples[idx] = @intCast((mid + side) >> 1);
                    samples[idx + 1] = @intCast((mid - side) >> 1);
                }
            },
            else => {},
        }

        frame_sample_offset += frame_header.channels.count() * block_size;
    }

    const sample_rate_hz: u24 = switch (sample_rate) {
        .StoredInMetadata => if (stream_info) |si| si.sample_rate else return error.InvalidFrameHeader,
        .Uncommon8b => try reader.readInt(u8, .big),
        .Uncommon16b => try reader.readInt(u16, .big),
        .Uncommon16bx10 => 10 * @as(u24, try reader.readInt(u16, .big)),
        .Forbidden => return error.InvalidFrameHeader,
        else => |sr| sr.hz(),
    };

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

    std.debug.print("samples: {d}\n", .{samples.len});

    if (!std.mem.eql(u8, &computed_md5, &stream_info.?.md5)) {
        log.err("Invalid checksum", .{});
        // return error.InvalidChecksum;
    }

    return .{
        .channels = channel_count,
        .sample_rate = sample_rate_hz,
        .bits_per_sample = bits_per_sample,
        .samples = samples,
    };
}

pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
    allocator.free(self.samples);
}

test "Example 1" {
    std.debug.print("---------- Example 1 ----------\n", .{});

    const Example = [_]u8{
        0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x0f, 0x00, 0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00,
        0x00, 0x01, 0x3e, 0x84, 0xb4, 0x18, 0x07, 0xdc, 0x69, 0x03, 0x07, 0x58,
        0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f, 0xff, 0xf8, 0x69, 0x18, 0x00, 0x00,
        0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa, 0x9a,
    };

    var file = std.io.fixedBufferStream(&Example);

    var r = try decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    try std.testing.expectEqual(2, r.channels);
    try std.testing.expectEqual(2, r.samples.len);
    try std.testing.expectEqual(25588, r.samples[0]);
    try std.testing.expectEqual(10416, r.samples[1]);
}

test "Example 2" {
    std.debug.print("---------- Example 2 ----------\n", .{});

    const Example = [_]u8{
        0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22, 0x00, 0x10, 0x00, 0x10,
        0x00, 0x00, 0x17, 0x00, 0x00, 0x44, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00,
        0x00, 0x13, 0xd5, 0xb0, 0x56, 0x49, 0x75, 0xe9, 0x8b, 0x8d, 0x8b, 0x93,
        0x04, 0x22, 0x75, 0x7b, 0x81, 0x03, 0x03, 0x00, 0x00, 0x12, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x04, 0x00, 0x00, 0x3a, 0x20, 0x00, 0x00, 0x00,
        0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x20, 0x6c, 0x69,
        0x62, 0x46, 0x4c, 0x41, 0x43, 0x20, 0x31, 0x2e, 0x33, 0x2e, 0x33, 0x20,
        0x32, 0x30, 0x31, 0x39, 0x30, 0x38, 0x30, 0x34, 0x01, 0x00, 0x00, 0x00,
        0x0e, 0x00, 0x00, 0x00, 0x54, 0x49, 0x54, 0x4c, 0x45, 0x3d, 0xd7, 0xa9,
        0xd7, 0x9c, 0xd7, 0x95, 0xd7, 0x9d, 0x81, 0x00, 0x00, 0x06, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xff, 0xf8, 0x69, 0x98, 0x00, 0x0f, 0x99, 0x12,
        0x08, 0x67, 0x01, 0x62, 0x3d, 0x14, 0x42, 0x99, 0x8f, 0x5d, 0xf7, 0x0d,
        0x6f, 0xe0, 0x0c, 0x17, 0xca, 0xeb, 0x21, 0x00, 0x0e, 0xe7, 0xa7, 0x7a,
        0x24, 0xa1, 0x59, 0x0c, 0x12, 0x17, 0xb6, 0x03, 0x09, 0x7b, 0x78, 0x4f,
        0xaa, 0x9a, 0x33, 0xd2, 0x85, 0xe0, 0x70, 0xad, 0x5b, 0x1b, 0x48, 0x51,
        0xb4, 0x01, 0x0d, 0x99, 0xd2, 0xcd, 0x1a, 0x68, 0xf1, 0xe6, 0xb8, 0x10,
        0xff, 0xf8, 0x69, 0x18, 0x01, 0x02, 0xa4, 0x02, 0xc3, 0x82, 0xc4, 0x0b,
        0xc1, 0x4a, 0x03, 0xee, 0x48, 0xdd, 0x03, 0xb6, 0x7c, 0x13, 0x30,
    };
    var file = std.io.fixedBufferStream(&Example);

    var r = try decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    try std.testing.expectEqual(2, r.channels);
    try std.testing.expectEqual(2 * (16 + 3), r.samples.len);
    try std.testing.expectEqualSlices(i16, &[_]i16{
        10372,                      6070,
        18041,                      10545,
        14942,                      8743,
        17876,                      10449,
        15627,                      9143,
        17899,                      10463,
        16242,                      9502,
        18077,                      10569,
        16824,                      9840,
        18263,                      10680,
        17295,                      10113,
        -14418,                     -8428,
        -15201,                     -8895,
        -14508,                     -8476,
        -15195,                     -8896,
        -14818,                     -8653,

        @bitCast(@as(u16, 0xc382)), @bitCast(@as(u16, 0b1101110010010000)),
        @bitCast(@as(u16, 0xc40b)), @bitCast(@as(u16, 0b1101110100000010)),
        @bitCast(@as(u16, 0xc14a)), @bitCast(@as(u16, 0b1101101100111110)),
    }, r.samples);
}

test "Example 3" {
    std.debug.print("---------- Example 3 ----------\n", .{});

    const Example = [_]u8{
        0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x1f, 0x00, 0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00,
        0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb, 0xcf, 0xc6, 0xdc, 0x80,
        0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
        0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb,
        0x6d, 0x09, 0x08, 0x31, 0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57,
        0xa3,
    };
    var file = std.io.fixedBufferStream(&Example);

    var r = try decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    try std.testing.expectEqual(1, r.channels);
    try std.testing.expectEqual(24, r.samples.len);
    try std.testing.expectEqualSlices(i16, &[_]i16{ 0, 79, 111, 78, 8, -61, -90, -68, -13, 42, 67, 53, 13, -27, -46, -38, -12, 14, 24, 19, 6, -4, -5, 0 }, r.samples);
}

fn run_standard_test(comptime filename: []const u8) !void {
    std.debug.print("---------- " ++ filename ++ " ----------\n", .{});

    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/subset/" ++ filename ++ ".flac", .{});
    defer file.close();

    var r = try decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    const expected = @embedFile("tests_expected_samples/" ++ filename ++ ".raw");
    try std.testing.expectEqualSlices(i16, @as([*]const i16, @alignCast(@ptrCast(expected)))[0 .. expected.len / 2], r.samples);
}

test "01 - blocksize 4096" {
    try run_standard_test("01 - blocksize 4096");
}

test "02 - blocksize 4608" {
    try run_standard_test("02 - blocksize 4608");
}

test "03 - blocksize 16" {
    try run_standard_test("03 - blocksize 16");
}

test "04 - blocksize 192" {
    try run_standard_test("04 - blocksize 192");
}

test "05 - blocksize 254" {
    try run_standard_test("05 - blocksize 254");
}

test "06 - blocksize 512" {
    try run_standard_test("06 - blocksize 512");
}

test "07 - blocksize 725" {
    try run_standard_test("07 - blocksize 725");
}

test "08 - blocksize 1000" {
    try run_standard_test("08 - blocksize 1000");
}

test "09 - blocksize 1937" {
    try run_standard_test("09 - blocksize 1937");
}

test "10 - blocksize 2304" {
    try run_standard_test("10 - blocksize 2304");
}

test "11 - partition order 8" {
    try run_standard_test("11 - partition order 8");
}

test "12 - qlp precision 15 bit" {
    try run_standard_test("12 - qlp precision 15 bit");
}

test "13 - qlp precision 2 bit" {
    try run_standard_test("13 - qlp precision 2 bit");
}

test "14 - wasted bits" {
    try run_standard_test("14 - wasted bits");
}

test "15 - only verbatim subframes" {
    try run_standard_test("15 - only verbatim subframes");
}

test "16 - partition order 8 containing escaped partitions" {
    try run_standard_test("16 - partition order 8 containing escaped partitions");
}

test "17 - all fixed orders" {
    try run_standard_test("17 - all fixed orders");
}

test "18 - precision search" {
    try run_standard_test("18 - precision search");
}

test "19 - samplerate 35467Hz" {
    try run_standard_test("19 - samplerate 35467Hz");
}

test "20 - samplerate 39kHz" {
    try run_standard_test("20 - samplerate 39kHz");
}

test "21 - samplerate 22050Hz" {
    try run_standard_test("21 - samplerate 22050Hz");
}

test "22 - 12 bit per sample" {
    try run_standard_test("22 - 12 bit per sample");
}

test "23 - 8 bit per sample" {
    try run_standard_test("23 - 8 bit per sample");
}

test "24 - variable blocksize file created with flake revision 264" {
    try run_standard_test("24 - variable blocksize file created with flake revision 264");
}

test "25 - variable blocksize file created with flake revision 264, modified to create smaller blocks" {
    try run_standard_test("25 - variable blocksize file created with flake revision 264, modified to create smaller blocks");
}

test "26 - variable blocksize file created with CUETools.Flake 2.1.6" {
    try run_standard_test("26 - variable blocksize file created with CUETools.Flake 2.1.6");
}

test "27 - old format variable blocksize file created with Flake 0.11" {
    try run_standard_test("27 - old format variable blocksize file created with Flake 0.11");
}

test "28 - high resolution audio, default settings" {
    try run_standard_test("28 - high resolution audio, default settings");
}

test "29 - high resolution audio, blocksize 16384" {
    try run_standard_test("29 - high resolution audio, blocksize 16384");
}

test "30 - high resolution audio, blocksize 13456" {
    try run_standard_test("30 - high resolution audio, blocksize 13456");
}

test "31 - high resolution audio, using only 32nd order predictors" {
    try run_standard_test("31 - high resolution audio, using only 32nd order predictors");
}

test "32 - high resolution audio, partition order 8 containing escaped partitions" {
    try run_standard_test("32 - high resolution audio, partition order 8 containing escaped partitions");
}

test "33 - samplerate 192kHz" {
    try run_standard_test("33 - samplerate 192kHz");
}

test "34 - samplerate 192kHz, using only 32nd order predictors" {
    try run_standard_test("34 - samplerate 192kHz, using only 32nd order predictors");
}

// test "35 - samplerate 134560Hz" {
//     try run_standard_test("35 - samplerate 134560Hz");
// }

// test "36 - samplerate 384kHz" {
//     try run_standard_test("36 - samplerate 384kHz");
// }

// test "37 - 20 bit per sample" {
//     try run_standard_test("37 - 20 bit per sample");
// }

// test "38 - 3 channels (3.0)" {
//    try run_standard_test("38 - 3 channels (3.0)");
// }

// test "39 - 4 channels (4.0)" {
//    try run_standard_test("39 - 4 channels (4.0)");
// }

// test "40 - 5 channels (5.0)" {
//    try run_standard_test("40 - 5 channels (5.0)");
// }

// test "41 - 6 channels (5.1)" {
//    try run_standard_test("41 - 6 channels (5.1)");
// }

// test "42 - 7 channels (6.1)" {
//    try run_standard_test("42 - 7 channels (6.1)");
// }

// test "43 - 8 channels (7.1)" {
//    try run_standard_test("43 - 8 channels (7.1)");
// }

// test "44 - 8-channel surround, 192kHz, 24 bit, using only 32nd order predictors" {
//    try run_standard_test("44 - 8-channel surround, 192kHz, 24 bit, using only 32nd order predictors");
// }

// test "45 - no total number of samples set" {
//    try run_standard_test("45 - no total number of samples set");
// }

// test "46 - no min-max framesize set" {
//    try run_standard_test("46 - no min-max framesize set");
// }

// test "47 - only STREAMINFO" {
//    try run_standard_test("47 - only STREAMINFO");
// }

// test "48 - Extremely large SEEKTABLE" {
//    try run_standard_test("48 - Extremely large SEEKTABLE");
// }

// test "49 - Extremely large PADDING" {
//    try run_standard_test("49 - Extremely large PADDING");
// }

// test "50 - Extremely large PICTURE" {
//    try run_standard_test("50 - Extremely large PICTURE");
// }

// test "51 - Extremely large VORBISCOMMENT" {
//    try run_standard_test("51 - Extremely large VORBISCOMMENT");
// }

// test "52 - Extremely large APPLICATION" {
//    try run_standard_test("52 - Extremely large APPLICATION");
// }

// test "53 - CUESHEET with very many indexes" {
//    try run_standard_test("53 - CUESHEET with very many indexes");
// }

// test "54 - 1000x repeating VORBISCOMMENT" {
//    try run_standard_test("54 - 1000x repeating VORBISCOMMENT");
// }

// test "55 - file 48-53 combined" {
//    try run_standard_test("55 - file 48-53 combined");
// }

// test "56 - JPG PICTURE" {
//    try run_standard_test("56 - JPG PICTURE");
// }

// test "57 - PNG PICTURE" {
//    try run_standard_test("57 - PNG PICTURE");
// }

// test "58 - GIF PICTURE" {
//    try run_standard_test("58 - GIF PICTURE");
// }

// test "59 - AVIF PICTURE" {
//    try run_standard_test("59 - AVIF PICTURE");
// }

// test "60 - mono audio" {
//    try run_standard_test("60 - mono audio");
// }

// test "61 - predictor overflow check, 16-bit" {
//    try run_standard_test("61 - predictor overflow check, 16-bit");
// }

// test "62 - predictor overflow check, 20-bit" {
//    try run_standard_test("62 - predictor overflow check, 20-bit");
// }

// test "63 - predictor overflow check, 24-bit" {
//    try run_standard_test("63 - predictor overflow check, 24-bit");
// }

// test "64 - rice partitions with escape code zero" {
//    try run_standard_test("64 - rice partitions with escape code zero");
// }
