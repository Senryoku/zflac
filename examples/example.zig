const std = @import("std");

const zflac = @import("zflac");
const zaudio = @import("zaudio");

pub const log_level: std.log.Level = .debug;

fn decode_standard_test(allocator: std.mem.Allocator, comptime filename: []const u8) !zflac.DecodedFLAC {
    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/subset/" ++ filename ++ ".flac", .{});
    defer file.close();

    return try zflac.decode(allocator, file.reader());
}

const PlayState = struct {
    file: *const zflac.DecodedFLAC,
    current_sample: usize,

    pub fn fill(self: *PlayState, output: *anyopaque, frame_count: u32) void {
        switch (self.file.sample_bit_size()) {
            8 => self._fill(i8, i16, output, frame_count),
            16 => self._fill(i16, i16, output, frame_count),
            24 => self._fill(i32, i32, output, frame_count),
            32 => self._fill(i32, i32, output, frame_count),
            else => unreachable,
        }
    }

    pub fn _fill(self: *PlayState, comptime SampleType: type, comptime OutputType: type, output: *anyopaque, frame_count: u32) void {
        var out: [*]OutputType = @ptrCast(@alignCast(output));
        const samples = self.file.samples(SampleType) catch unreachable;
        for (0..self.file.channels * frame_count) |i| {
            if (SampleType == i8 and OutputType == i16) {
                out[i] = @as(i16, @intCast(samples[self.current_sample])) * 256;
            } else {
                out[i] = @intCast(samples[self.current_sample]);
            }
            self.current_sample += 1;
            self.current_sample %= samples.len;
            if (self.current_sample == 0) {
                std.log.info("Looping back...", .{});
            }
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const r = try decode_standard_test(allocator, "37 - 20 bit per sample");
    defer r.deinit(allocator);

    zaudio.init(allocator);
    defer zaudio.deinit();

    var play_state: PlayState = .{
        .file = &r,
        .current_sample = 0,
    };

    var audio_device_config = zaudio.Device.Config.init(.playback);
    audio_device_config.sample_rate = r.sample_rate;
    audio_device_config.data_callback = audio_callback;
    audio_device_config.user_data = &play_state;
    audio_device_config.period_size_in_frames = 16;
    audio_device_config.playback.format = switch (r.sample_bit_size()) {
        8 => .signed16,
        16 => .signed16,
        24 => .signed32,
        32 => .signed32,
        else => return error.UnsupportedSampleBitSize,
    };
    audio_device_config.playback.channels = r.channels;

    var audio_device = try zaudio.Device.create(null, audio_device_config);
    audio_device.start() catch |err| {
        std.log.err("Failed to start audio device: {}", .{err});
        return;
    };

    std.log.info("Playing audio", .{});

    while (true) {}
}

fn audio_callback(
    device: *zaudio.Device,
    output: ?*anyopaque,
    _: ?*const anyopaque, // Input
    frame_count: u32,
) callconv(.C) void {
    const state: *PlayState = @ptrCast(@alignCast(device.getUserData()));

    if (output) |out| {
        state.fill(out, frame_count);
    }
}
