const std = @import("std");
const builtin = @import("builtin");

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

const FrameHeader = packed struct(u32) {
    zero: u1,
    bit_depth: BitDepth,
    channels: Channels,
    sample_rate: SampleRate,
    block_size: u4,
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

const low_bit_mask = [_]u8{
    0b00000000,
    0b00000001,
    0b00000011,
    0b00000111,
    0b00001111,
    0b00011111,
    0b00111111,
    0b01111111,
    // 0b11111111,
};

inline fn read_unary_integer_from_empty_buffer(bit_reader: anytype) !u32 {
    std.debug.assert(bit_reader.count == 0);
    if (try bit_reader.readBitsNoEof(u1, 1) == 0) { // Force fetching the next byte
        std.debug.assert(bit_reader.count == 7);
        var unary_integer: u32 = 1;
        const buffered_bits = (bit_reader.bits << @intCast(8 - bit_reader.count)) | low_bit_mask[8 - bit_reader.count];
        const clz = @clz(buffered_bits);
        unary_integer += clz;
        if (clz == bit_reader.count) {
            bit_reader.alignToByte(); // Discard those 0 bits
            while (try bit_reader.readBitsNoEof(u1, 1) == 0) unary_integer += 1; // Revert to simple version
        } else {
            _ = try bit_reader.readBitsNoEof(u8, clz + 1); // Discard those bits and the 1
        }
        return unary_integer;
    }
    return 0;
}

inline fn read_unary_integer(bit_reader: anytype) !u32 {
    if (false) {
        // Easy to read and portable version.
        var unary_integer: u32 = 0;
        while (try bit_reader.readBitsNoEof(u1, 1) == 0) unary_integer += 1;
        return unary_integer;
    } else {
        // WIP: Faster version relying on the internals of std.io.bitReader
        if (bit_reader.count == 0)
            return try read_unary_integer_from_empty_buffer(bit_reader);
        const buffered_bits = (bit_reader.bits << @intCast(8 - bit_reader.count)) | low_bit_mask[8 - bit_reader.count];
        const clz = @clz(buffered_bits);
        var unary_integer: u32 = clz;
        if (clz == bit_reader.count) {
            bit_reader.alignToByte(); // Discard those 0 bits
            if (false) {
                // Revert to simple version immediately
                while (try bit_reader.readBitsNoEof(u1, 1) == 0) unary_integer += 1;
            } else {
                // Inline a second round before reverting (NOTE: a recursive call would prevent inlining and seems to be slower)
                unary_integer += try read_unary_integer_from_empty_buffer(bit_reader);
            }
            return unary_integer;
        } else {
            _ = try bit_reader.readBitsNoEof(u8, clz + 1); // Discard those bits and the 1
            return unary_integer;
        }
    }
}

/// Reads a signed integer with a runtime known bit depth
inline fn read_signed_integer(comptime T: type, bit_reader: anytype, bit_depth: u6) !T {
    if (bit_depth == 0) return 0;
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

inline fn read_unencoded_sample(comptime SampleType: type, bit_reader: anytype, wasted_bits: u6, bits_per_sample: u6) !SampleType {
    const InterType = switch (SampleType) {
        i8 => i16,
        i16 => i32,
        i24 => i32,
        i32 => i64,
        else => @compileError("Unsupported sample type: " ++ @typeName(SampleType)),
    };
    if (wasted_bits > 0) {
        return @intCast(try read_signed_integer(InterType, bit_reader, bits_per_sample - wasted_bits));
    } else {
        return @intCast(try read_signed_integer(InterType, bit_reader, bits_per_sample));
    }
}

fn decode_residuals(comptime ResidualType: type, allocator: std.mem.Allocator, block_size: u32, order: u32, bit_reader: anytype) ![]ResidualType {
    var residuals = try allocator.alloc(ResidualType, block_size - order);
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
            if (bit_depth == 0) {
                @memset(residuals[partition_start_idx..][0..count], 0);
            } else {
                for (0..count) |i|
                    residuals[partition_start_idx + i] = try read_signed_integer(ResidualType, bit_reader, bit_depth);
            }
        } else {
            const UnsignedResidualType = std.meta.Int(.unsigned, @bitSizeOf(ResidualType));
            for (0..count) |i| {
                const quotient: UnsignedResidualType = @intCast(try read_unary_integer(bit_reader));
                const remainder = try bit_reader.readBitsNoEof(UnsignedResidualType, rice_parameter);
                const zigzag_encoded: UnsignedResidualType = (quotient << @intCast(rice_parameter)) + remainder;
                const residual: ResidualType = @bitCast((zigzag_encoded >> 1) ^ @as(UnsignedResidualType, @bitCast(-@as(ResidualType, @intCast(zigzag_encoded & 1)))));
                residuals[partition_start_idx + i] = residual;
            }
        }
        partition_start_idx += count;
    }

    return residuals;
}

pub fn decode(allocator: std.mem.Allocator, reader: anytype) !DecodedFLAC {
    const signature = try reader.readInt(u32, .big);
    if (signature != Signature)
        return error.InvalidSignature;

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
                    .channel_count = try bit_reader.readBitsNoEof(u3, 3), // NOTE: channel count - 1
                    .sample_bit_depth = try bit_reader.readBitsNoEof(u5, 5), // NOTE: bits per sample - 1
                    .number_of_samples = try bit_reader.readBitsNoEof(u36, 36),
                    .md5 = try reader.readBytesNoEof(16),
                };
                log.debug("  stream info: {any}", .{stream_info});
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

inline fn fixed_predictor(comptime SampleType: type, comptime InterType: type, comptime order: u6, block_size: u16, channel_count: u16, samples: []SampleType, residuals: []const InterType) void {
    // if (order == 4) return fixed_predictor_4(SampleType, InterType, block_size, channel_count, samples, residuals);
    if (order == 4) return linear_predictor_4(SampleType, InterType, block_size, channel_count, 0, [4]SampleType{ 4, -6, 4, -1 }, samples, residuals);
    for (0..block_size - order) |i| {
        const idx = channel_count * (order + i);
        samples[idx] = @intCast(switch (order) {
            0 => residuals[i],
            1 => residuals[i] + @as(InterType, 1) * samples[idx - channel_count * 1],
            2 => residuals[i] + @as(InterType, 2) * samples[idx - channel_count * 1] - @as(InterType, 1) * samples[idx - channel_count * 2],
            3 => residuals[i] + @as(InterType, 3) * samples[idx - channel_count * 1] - @as(InterType, 3) * samples[idx - channel_count * 2] + @as(InterType, 1) * samples[idx - channel_count * 3],
            4 => residuals[i] + @as(InterType, 4) * samples[idx - channel_count * 1] - @as(InterType, 6) * samples[idx - channel_count * 2] + @as(InterType, 4) * samples[idx - channel_count * 3] - samples[idx - channel_count * 4],
            else => unreachable,
        });
    }
}

inline fn linear_predictor_4(comptime SampleType: type, comptime InterType: type, block_size: u16, channel_count: u16, coefficient_shift_right: u6, predictor_coefficient: [4]SampleType, samples: []SampleType, residuals: []const InterType) void {
    const order = 4;
    var rolling_vector = @Vector(4, InterType){
        samples[channel_count * order - channel_count * 1],
        samples[channel_count * order - channel_count * 2],
        samples[channel_count * order - channel_count * 3],
        samples[channel_count * order - channel_count * 4],
    };
    const factors = [4]@Vector(4, InterType){
        .{ predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3] },
        .{ predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[0] },
        .{ predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[0], predictor_coefficient[1] },
        .{ predictor_coefficient[3], predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2] },
    };
    for (0..(block_size - order) / 4) |j| {
        const v_residuals = [4]@Vector(4, InterType){
            .{ 0, 0, 0, residuals[4 * j + 0] },
            .{ 0, 0, residuals[4 * j + 1], 0 },
            .{ 0, residuals[4 * j + 2], 0, 0 },
            .{ residuals[4 * j + 3], 0, 0, 0 },
        };
        inline for (0..4) |i| {
            rolling_vector[3 - (i % 4)] = @reduce(.Add, factors[i] * rolling_vector) >> @intCast(coefficient_shift_right);
            rolling_vector += v_residuals[i];
            const idx = channel_count * (order + 4 * j + i);
            samples[idx] = @intCast(rolling_vector[3 - (i % 4)]);
        }
    }
    for (0..(block_size - order) % 4) |i| {
        const idx = channel_count * (order + 4 * ((block_size - order) / 4) + i);
        const r = @reduce(.Add, factors[0] * @Vector(4, InterType){
            samples[idx - channel_count * 1],
            samples[idx - channel_count * 2],
            samples[idx - channel_count * 3],
            samples[idx - channel_count * 4],
        }) >> @intCast(coefficient_shift_right);
        samples[idx] = @intCast(r + residuals[4 * ((block_size - order) / 4) + i]);
    }
}

inline fn linear_predictor_8(comptime SampleType: type, comptime InterType: type, block_size: u16, channel_count: u16, coefficient_shift_right: u6, predictor_coefficient: [8]SampleType, samples: []SampleType, residuals: []const InterType) void {
    const order = 8;
    var rolling_vector = @Vector(order, InterType){
        samples[channel_count * order - channel_count * 1],
        samples[channel_count * order - channel_count * 2],
        samples[channel_count * order - channel_count * 3],
        samples[channel_count * order - channel_count * 4],
        samples[channel_count * order - channel_count * 5],
        samples[channel_count * order - channel_count * 6],
        samples[channel_count * order - channel_count * 7],
        samples[channel_count * order - channel_count * 8],
    };
    const factors = [order]@Vector(order, InterType){
        .{ predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[4], predictor_coefficient[5], predictor_coefficient[6], predictor_coefficient[7] },
        .{ predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[4], predictor_coefficient[5], predictor_coefficient[6], predictor_coefficient[7], predictor_coefficient[0] },
        .{ predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[4], predictor_coefficient[5], predictor_coefficient[6], predictor_coefficient[7], predictor_coefficient[0], predictor_coefficient[1] },
        .{ predictor_coefficient[3], predictor_coefficient[4], predictor_coefficient[5], predictor_coefficient[6], predictor_coefficient[7], predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2] },
        .{ predictor_coefficient[4], predictor_coefficient[5], predictor_coefficient[6], predictor_coefficient[7], predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3] },
        .{ predictor_coefficient[5], predictor_coefficient[6], predictor_coefficient[7], predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[4] },
        .{ predictor_coefficient[6], predictor_coefficient[7], predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[4], predictor_coefficient[5] },
        .{ predictor_coefficient[7], predictor_coefficient[0], predictor_coefficient[1], predictor_coefficient[2], predictor_coefficient[3], predictor_coefficient[4], predictor_coefficient[5], predictor_coefficient[6] },
    };
    for (0..(block_size - order) / order) |j| {
        // const v_residuals = [order]@Vector(order, InterType){
        //     .{ 0, 0, 0, 0, 0, 0, 0, residuals[order * j + 0] },
        //     .{ 0, 0, 0, 0, 0, 0, residuals[order * j + 1], 0 },
        //     .{ 0, 0, 0, 0, 0, residuals[order * j + 2], 0, 0 },
        //     .{ 0, 0, 0, 0, residuals[order * j + 3], 0, 0, 0 },
        //     .{ 0, 0, 0, residuals[order * j + 4], 0, 0, 0, 0 },
        //     .{ 0, 0, residuals[order * j + 5], 0, 0, 0, 0, 0 },
        //     .{ 0, residuals[order * j + 6], 0, 0, 0, 0, 0, 0 },
        //     .{ residuals[order * j + 7], 0, 0, 0, 0, 0, 0, 0 },
        // };
        inline for (0..order) |i| {
            rolling_vector[order - 1 - (i % order)] = @reduce(.Add, factors[i] * rolling_vector) >> @intCast(coefficient_shift_right);
            // rolling_vector += v_residuals[i];
            rolling_vector[order - 1 - (i % order)] += residuals[order * j + i];
            const idx = channel_count * (order + order * j + i);
            samples[idx] = @intCast(rolling_vector[order - 1 - (i % order)]);
        }
    }
    for (0..(block_size - order) % order) |i| {
        const idx = channel_count * (order + order * ((block_size - order) / order) + i);
        const r = @reduce(.Add, factors[0] * @Vector(order, InterType){
            samples[idx - channel_count * 1],
            samples[idx - channel_count * 2],
            samples[idx - channel_count * 3],
            samples[idx - channel_count * 4],
            samples[idx - channel_count * 5],
            samples[idx - channel_count * 6],
            samples[idx - channel_count * 7],
            samples[idx - channel_count * 8],
        }) >> @intCast(coefficient_shift_right);
        samples[idx] = @intCast(r + residuals[order * ((block_size - order) / order) + i]);
    }
}

inline fn linear_predictor(comptime SampleType: type, comptime InterType: type, comptime order: u6, block_size: u16, channel_count: u16, coefficient_shift_right: u6, predictor_coefficient: []const SampleType, samples: []SampleType, residuals: []const InterType) void {
    if (order == 4) return linear_predictor_4(SampleType, InterType, block_size, channel_count, coefficient_shift_right, predictor_coefficient[0..4].*, samples, residuals);
    if (order == 8) return linear_predictor_8(SampleType, InterType, block_size, channel_count, coefficient_shift_right, predictor_coefficient[0..8].*, samples, residuals);
    for (0..block_size - order) |i| {
        const idx = channel_count * (order + i);
        var predicted_without_shift: InterType = 0;
        for (0..order) |o| {
            predicted_without_shift += @as(InterType, @intCast(samples[idx - channel_count * (1 + o)])) * predictor_coefficient[o];
        }
        const predicted = predicted_without_shift >> @intCast(coefficient_shift_right);
        samples[idx] = @intCast(predicted + residuals[i]);
    }
}

fn decode_frames(comptime SampleType: type, allocator: std.mem.Allocator, stream_info: StreaminfoMetadata, reader: anytype) !DecodedFLAC {
    // Larger type for intermediate computations
    const InterType = switch (SampleType) {
        i8 => i16,
        i16 => i32,
        i24, i32 => i64,
        else => @compileError("Unsupported sample type: " ++ @typeName(SampleType)),
    };

    if (stream_info.number_of_samples == 0) return error.UnknownNumberOfSamples; // This is valid, but unsupported for now.

    var first_frame = true;
    var sample_rate: u24 = undefined;
    var channel_count: u4 = undefined;
    var bit_depth: BitDepth = undefined;
    var bits_per_sample: u6 = undefined;

    const expected_channel_count = @as(usize, stream_info.channel_count) + 1;
    const total_sample_count = expected_channel_count * stream_info.number_of_samples;
    const samples_backing = try allocator.allocWithOptions(u8, @sizeOf(SampleType) * total_sample_count, 16, null);
    errdefer allocator.free(samples_backing);

    var samples = @as([*]SampleType, @alignCast(@ptrCast(samples_backing.ptr)))[0 .. samples_backing.len / @sizeOf(SampleType)];

    var frame_sample_offset: usize = 0;
    // TODO: Get two bytes and check for FrameSync (0xFFF8 or 0xFFF9), rather than relying on knowing the number of samples in advance?
    while (frame_sample_offset < samples.len) {
        const frame_header: FrameHeader = @bitCast(try reader.readInt(u32, .big));
        log_frame.debug("frame header: {any} (frame_sample_offset: {d})", .{ frame_header, frame_sample_offset });
        if (frame_header.frame_sync != (0xFFF8 >> 1))
            return error.InvalidFrameHeader;

        const coded_number = try read_coded_number(&reader);
        switch (frame_header.blocking_strategy) {
            .Fixed => log_frame.debug("  Frame number: {d}", .{coded_number}),
            .Variable => log_frame.debug("  Sample number: {d}", .{coded_number}),
        }

        var bit_reader = std.io.bitReader(.big, reader);

        const block_size: u16 = switch (frame_header.block_size) {
            0b0000 => return error.InvalidFrameHeader, // Reserved
            0b0001 => 192,
            0b0010...0b0101 => |b| 144 * std.math.pow(u16, 2, b),
            // Uncommon block size
            0b0110 => @as(u16, try reader.readInt(u8, .big)) + 1,
            0b0111 => bs: {
                const ubs = try reader.readInt(u16, .big);
                if (ubs == std.math.maxInt(u16)) return error.InvalidFrameHeader;
                break :bs ubs + 1;
            },
            0b1000...0b1111 => |b| std.math.pow(u16, 2, b),
        };
        log_frame.debug("  block_size: {d}", .{block_size});

        const frame_sample_rate: u24 = switch (frame_header.sample_rate) {
            .StoredInMetadata => stream_info.sample_rate,
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
                .StoredInMetadata => @as(u6, stream_info.sample_bit_depth) + 1,
                else => |bd| bd.bps(),
            };

            if (channel_count != expected_channel_count) return error.InconsistentParameters;

            first_frame = false;
        } else {
            // "Because not all environments in which FLAC decoders are used are able to cope with changes to these properties during playback, a decoder MAY choose to stop decoding on such a change."
            if (sample_rate != frame_sample_rate or channel_count != frame_header.channels.count() or bit_depth != frame_header.bit_depth) return error.InconsistentParameters;

            const expected_samples = frame_sample_offset + @as(usize, block_size) * channel_count;
            if (samples.len < expected_samples) {
                return error.InvalidSampleCount;
                // We could support reallocating the backing buffer (when the total number of samples is unknown, for streaming I guess):
                //   samples_backing = try allocator.realloc(samples_backing, expected_samples);
                //   samples = @as([*]SampleType, @alignCast(@ptrCast(samples_backing.ptr)))[0 .. samples_backing.len / @sizeOf(SampleType)];
                // However, we currently rely on the total number of samples to stop processing.
            }
        }

        // Block size of 1 not allowed except for the last frame.
        if (block_size == 1 and frame_sample_offset + channel_count * block_size < samples.len) return error.InvalidFrameHeader;

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

            const unencoded_samples_bit_depth = switch (frame_header.channels) {
                .LRLeftSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                .LRSideRightStereo => if (channel == 0) bits_per_sample + 1 else bits_per_sample,
                .LRMidSideStereo => if (channel == 1) bits_per_sample + 1 else bits_per_sample,
                else => bits_per_sample,
            };

            switch (subframe_header.subframe_type) {
                0b000000 => { // Constant subframe
                    const sample = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, bits_per_sample);
                    if (channel_count == 1) {
                        @memset(samples[frame_sample_offset..][0..block_size], sample);
                    } else {
                        for (0..block_size) |i| {
                            samples[frame_sample_offset + channel_count * i + channel] = sample;
                        }
                    }
                },
                0b000001 => { // Verbatim subframe
                    for (0..block_size) |i| {
                        samples[frame_sample_offset + channel_count * i + channel] = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, unencoded_samples_bit_depth);
                        log_subframe.debug("    sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
                    }
                },
                0b001000...0b001100 => |t| { // Subframe with a fixed predictor of order v-8; i.e., 0, 1, 2, 3 or 4
                    const order: u3 = @intCast(t & 0b000111);
                    log_subframe.debug("  Subframe with a fixed predictor of order {d}", .{order});
                    if (order > 4) return error.InvalidSubframeHeader;

                    for (0..order) |i| {
                        samples[frame_sample_offset + channel_count * i + channel] = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, unencoded_samples_bit_depth);
                        log_subframe.debug("    warmup_sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
                    }

                    const residuals = try decode_residuals(InterType, allocator, block_size, order, &bit_reader);
                    defer allocator.free(residuals);

                    switch (order) {
                        5...7 => unreachable,
                        inline else => |comptime_order| fixed_predictor(SampleType, InterType, comptime_order, block_size, channel_count, samples[frame_sample_offset + channel ..], residuals),
                    }
                },
                0b100000...0b111111 => |t| { // Subframe with a linear predictor of order v-31; i.e., 1 through 32 (inclusive)
                    const order: u6 = @intCast(t - 31);
                    log_subframe.debug("  Subframe with a linear predictor of order: {d}", .{order});
                    // Unencoded warm-up samples (n = subframe's bits per sample * LPC order).
                    for (0..order) |i| {
                        samples[frame_sample_offset + channel_count * i + channel] = try read_unencoded_sample(SampleType, &bit_reader, wasted_bits, unencoded_samples_bit_depth);
                        log_subframe.debug("    warmup_sample: {d}", .{samples[frame_sample_offset + channel_count * i + channel]});
                    }
                    // (Predictor coefficient precision in bits)-1 (Note: 0b1111 is forbidden).
                    const coefficient_precision = (try bit_reader.readBitsNoEof(u4, 4)) + 1;
                    log_subframe.debug("    coefficient_precision: {d}", .{coefficient_precision});
                    // Prediction right shift needed in bits.
                    const coefficient_shift_right = try bit_reader.readBitsNoEof(u5, 5);
                    log_subframe.debug("    coefficient_shift_right: {d}", .{coefficient_shift_right});

                    // Predictor coefficients (n = predictor coefficient precision * LPC order).
                    var predictor_coefficient: [32]SampleType = undefined;
                    for (0..order) |i| {
                        predictor_coefficient[i] = try read_unencoded_sample(SampleType, &bit_reader, 0, coefficient_precision);
                        log_subframe.debug("    predictor_coefficient[{d}]: {d}", .{ i, predictor_coefficient[i] });
                    }

                    const residuals = try decode_residuals(InterType, allocator, block_size, order, &bit_reader);
                    defer allocator.free(residuals);

                    switch (order) {
                        33...63 => unreachable,
                        inline else => |comptime_order| {
                            linear_predictor(SampleType, InterType, comptime_order, block_size, channel_count, coefficient_shift_right, predictor_coefficient[0..order], samples[frame_sample_offset + channel ..], residuals);
                        },
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

    return .{
        .channels = channel_count,
        .sample_rate = sample_rate,
        .bits_per_sample = bits_per_sample,
        ._samples = samples_backing,
    };
}
