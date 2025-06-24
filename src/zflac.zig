const std = @import("std");
const builtin = @import("builtin");
const BitReader = @import("bit_reader.zig");

const log = std.log.scoped(.zflac);
const log_frame = std.log.scoped(.zflac_frame);
const log_subframe = std.log.scoped(.zflac_subframe);
const log_residual = std.log.scoped(.zflac_residual);

const Signature: u32 = 0x664C6143;

pub const DecodedFLAC = struct {
    channels: u8,
    sample_rate: u24,
    bits_per_sample: u8,
    _samples: []align(16) u8,

    pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self._samples);
    }

    pub fn sample_bit_size(self: @This()) u8 {
        return std.mem.alignForward(u8, self.bits_per_sample, 8);
    }

    pub fn samples(self: @This(), comptime SampleType: type) ![]SampleType {
        const expected_bit_size = self.sample_bit_size();
        const container_bit_size = try std.math.ceilPowerOfTwo(u8, expected_bit_size);
        if (@bitSizeOf(SampleType) != container_bit_size) return error.UnexpectedSampleType;
        return @as([*]SampleType, @alignCast(@ptrCast(self._samples.ptr)))[0 .. self._samples.len / (container_bit_size / 8)];
    }
};

const MetadataHeader = packed struct {
    info: enum(u7) {
        Streaminfo = 0,
        Padding = 1,
        Application = 2,
        Seektable = 3,
        VorbisComment = 4,
        Cuesheet = 5,
        Picture = 6,
        _,
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

const BlockSize = enum(u4) {
    Reserved = 0b0000,
    @"192" = 0b0001,
    Uncommon8b = 0b0110,
    Uncommon16b = 0b0111,
    _,

    pub fn value(self: @This()) u16 {
        return switch (@intFromEnum(self)) {
            0b0001 => 192,
            0b0010...0b0101 => |b| 144 * std.math.pow(u16, 2, b),
            0b1000...0b1111 => |b| std.math.pow(u16, 2, b),
            else => unreachable,
        };
    }
};

const FrameHeader = packed struct(u32) {
    zero: u1,
    bit_depth: BitDepth,
    channels: Channels,
    sample_rate: SampleRate,
    block_size: BlockSize,
    blocking_strategy: enum(u1) { Fixed = 0, Variable = 1 },
    frame_sync: u15,
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

/// Reads a signed integer with a runtime known bit depth
inline fn read_signed_integer(comptime T: type, bit_reader: anytype, bit_depth: u6) !T {
    std.debug.assert(bit_depth > 0 and bit_depth <= @bitSizeOf(T));
    const ContainerType = std.meta.Int(.unsigned, @bitSizeOf(T));
    var r = try bit_reader.readBitsNoEof(ContainerType, bit_depth);
    // Sign extend from bit_depth to container_type size
    const shift = @bitSizeOf(ContainerType) - @as(usize, bit_depth);
    r <<= @intCast(shift);
    return @as(T, @bitCast(r)) >> @intCast(shift);
}

inline fn read_unencoded_sample(comptime SampleType: type, bit_reader: anytype, wasted_bits: u6, bits_per_sample: u6) !SampleType {
    const InterType = std.meta.Int(.signed, try std.math.ceilPowerOfTwo(u32, @bitSizeOf(SampleType) + 1));
    return @intCast(try read_signed_integer(InterType, bit_reader, bits_per_sample - wasted_bits));
}

fn read_coded_number(reader: anytype) !u64 {
    const first_byte = try reader.readByte();
    const byte_count = @clz(first_byte ^ 0xFF);
    if (first_byte == 0xFF or byte_count == 1) return error.InvalidCodedNumber;
    if (byte_count == 0) return first_byte;
    var coded_number: u64 = (first_byte & (@as(u8, 0x7F) >> @intCast(byte_count)));
    for (0..byte_count - 1) |_| {
        coded_number <<= 6;
        coded_number |= (try reader.readByte()) & 0x3F;
    }
    return coded_number;
}

pub fn decode(allocator: std.mem.Allocator, reader: anytype) !DecodedFLAC {
    const signature = try reader.readInt(u32, .big);
    if (signature != Signature)
        return error.InvalidSignature;

    var stream_info: ?StreaminfoMetadata = null;
    while (true) {
        var header = try reader.readStruct(MetadataHeader);
        header.length = @byteSwap(header.length);

        switch (header.info) {
            .Streaminfo => {
                var bit_reader = BitReader.init(reader);
                stream_info = .{
                    .min_block_size = try bit_reader.readBitsNoEof(u16, 16),
                    .max_block_size = try bit_reader.readBitsNoEof(u16, 16),
                    .min_frame_size = try bit_reader.readBitsNoEof(u24, 24),
                    .max_frame_size = try bit_reader.readBitsNoEof(u24, 24),
                    .sample_rate = try bit_reader.readBitsNoEof(u20, 20),
                    .channel_count = try bit_reader.readBitsNoEof(u3, 3), // NOTE: channel count - 1
                    .sample_bit_depth = try bit_reader.readBitsNoEof(u5, 5), // NOTE: bits per sample - 1
                    .number_of_samples = try bit_reader.readBitsNoEof(u36, 36),
                    .md5 = try reader.readBytesNoEof(16),
                };
                log.debug("{any}", .{stream_info});
            },
            .Application, .Seektable, .VorbisComment, .Cuesheet, .Picture => {
                log.info(" Unhandled header: {s}", .{@tagName(header.info)});
                try reader.skipBytes(header.length, .{});
            },
            .Padding => try reader.skipBytes(header.length, .{}),
            else => return error.InvalidMetadataHeader,
        }

        if (header.last_block)
            break;
    }

    if (stream_info) |si| {
        const sample_bit_depth = @as(u8, si.sample_bit_depth) + 1;
        const aligned_sample_bit_size: u8 = std.mem.alignForward(u8, sample_bit_depth, 8);

        const decoded = try switch (aligned_sample_bit_size) {
            8 => decode_frames(i8, allocator, si, reader),
            16 => decode_frames(i16, allocator, si, reader),
            24 => decode_frames(i32, allocator, si, reader),
            32 => decode_frames(i32, allocator, si, reader),
            else => return error.Unimplemented,
        };
        errdefer decoded.deinit(allocator);

        var computed_md5: [16]u8 = undefined;
        if (aligned_sample_bit_size == 24) {
            // While zig does support i24, i24 in arrays are still 4 bytes aligned. Might as well use i32.
            const samples_32 = try decoded.samples(i32);
            var d = std.crypto.hash.Md5.init(.{});
            for (0..samples_32.len) |i| {
                d.update(std.mem.asBytes(&samples_32[i])[0..3]);
            }
            d.final(&computed_md5);
        } else {
            std.crypto.hash.Md5.hash(decoded._samples, &computed_md5, .{});
        }

        if (!std.mem.eql(u8, &computed_md5, &stream_info.?.md5))
            return error.InvalidChecksum;

        // The MD5 checksum is computed on sign extended values (12 to 16 for example), however it seems
        // output conventions differ quite a bit. I don't known if I should do this here, or let the caller deal with it.
        //   ( - Use unsigned u8 for 8bits per samples. )
        //   - Up 12bits samples to 16bits by multiplying by 16.
        //   - Up 24bits samples to 32bits by multiplying by 256.
        switch (sample_bit_depth) {
            // 8 => {
            //     for (0..decoded._samples.len) |i| {
            //         decoded._samples[i] +%= 128;
            //     }
            // },
            9...15 => |bd| {
                var samples_16 = try decoded.samples(i16);
                for (0..samples_16.len) |i| {
                    samples_16[i] <<= @intCast(16 - bd);
                }
            },
            17...31 => |bd| {
                var samples_32 = try decoded.samples(i32);
                for (0..samples_32.len) |i| {
                    samples_32[i] <<= @intCast(@as(u6, 32) - bd);
                }
            },
            else => {},
        }

        return decoded;
    } else return error.MissingStreaminfo;
}

fn decode_frames(comptime SampleType: type, allocator: std.mem.Allocator, stream_info: StreaminfoMetadata, reader: anytype) !DecodedFLAC {
    // Larger type for intermediate computations
    const InterType = switch (SampleType) {
        i8 => i16,
        i16 => i32,
        i24, i32 => i64,
        else => @compileError("Unsupported sample type: " ++ @typeName(SampleType)),
    };

    var first_frame = true;
    var sample_rate: u24 = undefined;
    var channel_count: u4 = undefined;
    var bit_depth: BitDepth = undefined;
    var bits_per_sample: u6 = undefined;

    var valid_total_sample_count = stream_info.number_of_samples > 0;

    const expected_channel_count = @as(usize, stream_info.channel_count) + 1;
    const total_sample_count = expected_channel_count * (if (valid_total_sample_count) stream_info.number_of_samples else 4096);
    var samples_backing = try allocator.allocWithOptions(u8, @sizeOf(SampleType) * total_sample_count, 16, null);
    errdefer allocator.free(samples_backing);

    var samples = @as([*]SampleType, @alignCast(@ptrCast(samples_backing.ptr)))[0 .. samples_backing.len / @sizeOf(SampleType)];

    var samples_working_buffer = try allocator.alloc(InterType, if (stream_info.max_block_size > 0) stream_info.max_block_size else 4096);
    defer allocator.free(samples_working_buffer);

    var frame_sample_offset: usize = 0;
    while (true) {
        if (valid_total_sample_count and frame_sample_offset >= total_sample_count) break;

        const frame_header: FrameHeader = @bitCast(reader.readInt(u32, .big) catch |err| {
            if (valid_total_sample_count) return err; // Unexpected EOF, based on the expected number of samples from the metadata.
            // Unknown number of samples: EndOfStream on frame boundary isn't necessarily an error.
            switch (err) {
                error.EndOfStream => break,
                else => |e| return e,
            }
        });
        if (frame_header.frame_sync != (0xFFF8 >> 1))
            return error.InvalidFrameHeader; // NOTE: We could try to return normally when valid_total_sample_count is false here. The CRC check should catch if this was the wrong decision.

        const coded_number = try read_coded_number(&reader);

        const block_size: u16 = switch (frame_header.block_size) {
            .Reserved => return error.InvalidFrameHeader,
            .Uncommon8b => @as(u16, try reader.readInt(u8, .big)) + 1,
            .Uncommon16b => bs: {
                const ubs = try reader.readInt(u16, .big);
                if (ubs == std.math.maxInt(u16)) return error.InvalidFrameHeader;
                break :bs ubs + 1;
            },
            else => |b| b.value(),
        };

        const frame_sample_rate: u24 = switch (frame_header.sample_rate) {
            .StoredInMetadata => stream_info.sample_rate,
            .Uncommon8b => try reader.readInt(u8, .big),
            .Uncommon16b => try reader.readInt(u16, .big),
            .Uncommon16bx10 => 10 * @as(u24, try reader.readInt(u16, .big)),
            .Forbidden => return error.InvalidFrameHeader,
            else => |sr| sr.hz(),
        };

        if (first_frame) {
            sample_rate = frame_sample_rate;
            channel_count = frame_header.channels.count();
            bit_depth = frame_header.bit_depth;

            bits_per_sample = switch (frame_header.bit_depth) {
                .StoredInMetadata => @as(u6, stream_info.sample_bit_depth) + 1,
                else => |bd| bd.bps(),
            };

            if (channel_count != expected_channel_count) return error.InconsistentParameters;

            first_frame = false;
        } else {
            // "Because not all environments in which FLAC decoders are used are able to cope with changes to these properties during playback, a decoder MAY choose to stop decoding on such a change."
            if (sample_rate != frame_sample_rate or channel_count != frame_header.channels.count() or bit_depth != frame_header.bit_depth) return error.InconsistentParameters;
        }

        const expected_samples = frame_sample_offset + @as(usize, block_size) * channel_count;
        if (samples.len < expected_samples) {
            // Since this should only happen when the number of samples is unknown, or wrong, increase it more than strictly necessary since we'd have to at each new frame otherwise.
            // The buffer will be trimmed to the correct size once the file has been fully processed.
            const new_size = @max(2 * samples.len, expected_samples);
            samples_backing = try allocator.realloc(samples_backing, new_size * @sizeOf(SampleType));
            samples = @as([*]SampleType, @alignCast(@ptrCast(samples_backing.ptr)))[0 .. samples_backing.len / @sizeOf(SampleType)];
            valid_total_sample_count = false; // We now know the number of samples from the metadata was wrong, we can't rely on it to stop processing the file.
        }

        // Block size of 1 not allowed except for the very last frame.
        if (block_size == 1 and (valid_total_sample_count and frame_sample_offset + channel_count * block_size < total_sample_count)) return error.InvalidFrameHeader;

        const frame_header_crc = try reader.readInt(u8, .big);
        // TODO: Check CRC
        // Finally, an 8-bit CRC follows the frame/sample number, an uncommon block size, or an uncommon sample rate (depending on whether the latter two are stored).
        // This CRC is initialized with 0 and has the polynomial x^8 + x^2 + x^1 + x^0. This CRC covers the whole frame header before the CRC, including the sync code.

        log_frame.debug("Frame: First sample: {d}, Sample Rate: {s} ({d}Hz), Channels: {s} ({d}), Bit Depth: {s} ({d}), Block Size: {s} ({d}), CRC: {X:0>2}, Coded Number ({s}): {d}", .{
            frame_sample_offset, @tagName(frame_header.sample_rate), frame_sample_rate, @tagName(frame_header.channels),
            channel_count,       @tagName(frame_header.bit_depth),   bits_per_sample,   std.enums.tagName(BlockSize, frame_header.block_size) orelse "Common",
            block_size,          frame_header_crc,
            switch (frame_header.blocking_strategy) {
                .Fixed => "Frame number",
                .Variable => "Sample number",
            },
            coded_number,
        });

        var bit_reader = BitReader.init(reader);

        for (0..channel_count) |channel| {
            const subframe_header: SubframeHeader = .{
                .zero = try bit_reader.readBitsNoEof(u1, 1),
                .subframe_type = try bit_reader.readBitsNoEof(u6, 6),
                .wasted_bit_flag = try bit_reader.readBitsNoEof(u1, 1),
            };
            if (subframe_header.zero != 0) return error.InvalidSubframeHeader;

            const wasted_bits: u6 = if (subframe_header.wasted_bit_flag == 1) @intCast(try bit_reader.readUnary() + 1) else 0;

            // "When stereo decorrelation is used, the side channel will have one extra bit of bit depth"
            const unencoded_samples_bit_depth = switch (frame_header.channels) {
                .LRLeftSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                .LRSideRightStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                .LRMidSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                else => bits_per_sample,
            };

            const subframe_samples = samples[frame_sample_offset..][channel .. @as(usize, channel_count) * block_size];
            switch (subframe_header.subframe_type) {
                0b000000 => { // Constant subframe
                    log_subframe.debug("Subframe #{d}: Constant, {d} wasted bits", .{ channel, wasted_bits });
                    const sample = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, bits_per_sample);
                    if (channel_count == 1) {
                        @memset(subframe_samples, sample);
                    } else {
                        for (0..block_size) |i|
                            subframe_samples[channel_count * i] = sample;
                    }
                },
                0b000001 => { // Verbatim subframe
                    log_subframe.debug("Subframe #{d}: Verbatim subframe, {d} wasted bits", .{ channel, wasted_bits });
                    for (0..block_size) |i|
                        subframe_samples[channel_count * i] = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, unencoded_samples_bit_depth);
                },
                0b001000...0b001100 => |t| { // Subframe with a fixed predictor of order v-8; i.e., 0, 1, 2, 3 or 4
                    if (samples_working_buffer.len < block_size)
                        samples_working_buffer = try allocator.realloc(samples_working_buffer, block_size);

                    const order: u3 = @intCast(t & 0b000111);
                    if (order > 4) return error.InvalidSubframeHeader;
                    // Unencoded warm-up samples (n = subframe's bits per sample * LPC order).
                    for (0..order) |i|
                        samples_working_buffer[i] = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, unencoded_samples_bit_depth);

                    log_subframe.debug("Subframe #{d}: Fixed predictor of order {d}, {d} wasted bits", .{ channel, order, wasted_bits });
                    log_subframe.debug("  Warmup Samples: {d}", .{samples_working_buffer[0..order]});

                    try decode_residuals(InterType, samples_working_buffer[order..], block_size, order, &bit_reader);

                    for (order..block_size) |i| {
                        samples_working_buffer[i] += switch (order) {
                            0 => 0, // Just the residuals
                            1 => 1 * samples_working_buffer[i - 1],
                            2 => 2 * samples_working_buffer[i - 1] - 1 * samples_working_buffer[i - 2],
                            3 => 3 * samples_working_buffer[i - 1] - 3 * samples_working_buffer[i - 2] + 1 * samples_working_buffer[i - 3],
                            4 => 4 * samples_working_buffer[i - 1] - 6 * samples_working_buffer[i - 2] + 4 * samples_working_buffer[i - 3] - samples_working_buffer[i - 4],
                            else => unreachable,
                        };
                    }

                    // Interleave
                    for (0..block_size) |i|
                        subframe_samples[channel_count * i] = @intCast(samples_working_buffer[i]);
                },
                0b100000...0b111111 => |t| { // Subframe with a linear predictor of order v-31; i.e., 1 through 32 (inclusive)
                    if (samples_working_buffer.len < block_size)
                        samples_working_buffer = try allocator.realloc(samples_working_buffer, block_size);

                    const order: u6 = @intCast(t - 31);
                    // Unencoded warm-up samples (n = subframe's bits per sample * LPC order).
                    for (0..order) |i|
                        samples_working_buffer[i] = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, unencoded_samples_bit_depth);
                    // (Predictor coefficient precision in bits)-1 (Note: 0b1111 is forbidden).
                    const coefficient_precision = (try bit_reader.readBitsNoEof(u4, 4)) + 1;
                    // Prediction right shift needed in bits.
                    const prediction_shift_right = try bit_reader.readBitsNoEof(u5, 5);
                    // Predictor coefficients (n = predictor coefficient precision * LPC order).
                    var predictor_coefficient: [32]InterType = @splat(0);
                    for (0..order) |i| // Stored in reverse order to match the layout of samples in memory (samples_working_buffer).
                        predictor_coefficient[order - 1 - i] = try read_unencoded_sample(InterType, &bit_reader, 0, coefficient_precision);

                    log_subframe.debug("Subframe #{d}: Linear predictor of order {d}, {d} bits coefficients, {d} bits right shift, {d} wasted bits", .{ channel, order, coefficient_precision, prediction_shift_right, wasted_bits });
                    log_subframe.debug("  Warmup Samples: {d}", .{samples_working_buffer[0..order]});
                    log_subframe.debug("  Predictor Coefficients: {d}", .{predictor_coefficient[0..order]});

                    try decode_residuals(InterType, samples_working_buffer[order..], block_size, order, &bit_reader);

                    switch (order) {
                        0 => {}, // Just the residuals
                        33...63 => unreachable,
                        inline 8, 16, 24, 32 => |comptime_order| linear_predictor(InterType, comptime_order, block_size, prediction_shift_right, predictor_coefficient[0..comptime_order], samples_working_buffer),
                        inline else => |comptime_order| {
                            for (comptime_order..block_size) |i| {
                                var predicted_without_shift: InterType = 0;
                                for (0..comptime_order) |o|
                                    predicted_without_shift += @as(InterType, @intCast(samples_working_buffer[i - comptime_order + o])) * predictor_coefficient[o];
                                const predicted = predicted_without_shift >> @intCast(prediction_shift_right);
                                samples_working_buffer[i] += predicted;
                            }
                        },
                    }
                    // Interleave
                    for (0..block_size) |i|
                        subframe_samples[channel_count * i] = @intCast(samples_working_buffer[i]);
                },
                0b000010...0b000111, 0b001101...0b011111 => return error.InvalidSubframeHeader, // Reserved
            }

            if (wasted_bits > 0) {
                for (0..block_size) |i|
                    subframe_samples[channel_count * i] <<= @intCast(wasted_bits);
            }
        }
        // NOTE: Last subframe is padded with zero bits to be byte aligned.
        bit_reader.alignToByte();

        const frame_crc = try reader.readInt(u16, .big);
        _ = frame_crc;
        // TODO: Check CRC
        // "Following this is a 16-bit CRC, initialized with 0, with the polynomial x^16 + x^15 + x^2 + x^0. This CRC covers the whole frame, excluding the 16-bit CRC but including the sync code."

        // Stereo decorrelation
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
                    var mid = @as(InterType, samples[idx]) << 1;
                    const side = samples[idx + 1];
                    mid += side & 1;
                    samples[idx] = @intCast((mid + side) >> 1);
                    samples[idx + 1] = @intCast((mid - side) >> 1);
                }
            },
            else => {},
        }

        frame_sample_offset += @as(usize, @intCast(channel_count)) * block_size;
    }

    if (samples.len != frame_sample_offset) {
        // This should only be possible when the number of samples is unknown (absent from the metadata), or wrong.
        std.debug.assert(!valid_total_sample_count);
        samples_backing = try allocator.realloc(samples_backing, frame_sample_offset * @sizeOf(SampleType));
        samples = @as([*]SampleType, @alignCast(@ptrCast(samples_backing.ptr)))[0..frame_sample_offset];
    }

    return .{
        .channels = channel_count,
        .sample_rate = sample_rate,
        .bits_per_sample = bits_per_sample,
        ._samples = samples_backing,
    };
}

/// samples: order unencoded warmup samples followed by residuals
inline fn linear_predictor(comptime InterType: type, comptime order: u6, block_size: u16, prediction_shift_right: u6, predictor_coefficient: []const InterType, samples: []InterType) void {
    const pred_vector: @Vector(order, InterType) = predictor_coefficient[0..order].*;
    for (0..block_size - order) |i| {
        const s: @Vector(order, InterType) = samples[i..][0..order].*;
        const predicted_without_shift = @reduce(.Add, pred_vector * s);
        const predicted = predicted_without_shift >> @intCast(prediction_shift_right);
        samples[order + i] += predicted;
    }
}

fn decode_residuals(comptime ResidualType: type, residuals: []ResidualType, block_size: u16, order: u6, bit_reader: anytype) !void {
    std.debug.assert(residuals.len >= block_size - order);

    const coding_method = try bit_reader.readBitsNoEof(u2, 2);
    if (coding_method >= 0b10) return error.InvalidResidualCodingMethod;
    const partition_order = try bit_reader.readBitsNoEof(u4, 4);

    log_residual.debug("    Residual decoding: Coding method: {s}, Partition order: {d}", .{ if (coding_method == 0b00) "Rice" else "Rice2", partition_order });

    var partition_start_idx: u32 = 0;
    for (0..std.math.pow(u32, 2, partition_order)) |partition| {
        var count = (block_size >> partition_order);
        if (partition == 0) count -= order;
        switch (coding_method) {
            inline 0b00, 0b01 => |comptime_coding_method| try decode_residual_partition(ResidualType, @enumFromInt(comptime_coding_method), residuals[partition_start_idx..][0..count], bit_reader),
            else => unreachable,
        }
        partition_start_idx += count;
    }
}

fn decode_residual_partition(comptime ResidualType: type, comptime coding_method: enum(u2) { Rice = 0, Rice2 = 1 }, residuals: []ResidualType, bit_reader: anytype) !void {
    const UnsignedResidualType = std.meta.Int(.unsigned, @bitSizeOf(ResidualType));
    const RiceParameterType = switch (coding_method) {
        .Rice => u4,
        .Rice2 => u5,
    };

    const rice_parameter = try bit_reader.readBitsNoEof(RiceParameterType, @bitSizeOf(RiceParameterType));
    log_residual.debug("      Partition: Rice parameter {d}", .{rice_parameter});

    switch (rice_parameter) {
        std.math.maxInt(RiceParameterType) => { // No rice parameter
            const bit_depth: u5 = try bit_reader.readBitsNoEof(u5, 5);
            if (bit_depth == 0) {
                @memset(residuals, 0);
            } else {
                for (0..residuals.len) |i|
                    residuals[i] = try read_signed_integer(ResidualType, bit_reader, bit_depth);
            }
        },
        inline else => |comptime_rice_parameter| {
            if (comptime_rice_parameter >= @bitSizeOf(UnsignedResidualType)) unreachable;
            for (0..residuals.len) |i| {
                const quotient: UnsignedResidualType = @intCast(try bit_reader.readUnary());
                const remainder = try bit_reader.readBitsNoEof(UnsignedResidualType, comptime_rice_parameter);
                const zigzag_encoded: UnsignedResidualType = (quotient << @intCast(comptime_rice_parameter)) + remainder;
                const residual: ResidualType = @bitCast((zigzag_encoded >> 1) ^ @as(UnsignedResidualType, @bitCast(-@as(ResidualType, @intCast(zigzag_encoded & 1)))));
                residuals[i] = residual;
            }
        },
    }
}
