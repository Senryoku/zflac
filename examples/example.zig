const std = @import("std");

const zflac = @import("zflac");
const zaudio = @import("zaudio");

pub const log_level: std.log.Level = .debug;

fn decode_standard_test(allocator: std.mem.Allocator, comptime filename: []const u8) !zflac.DecodedFLAC {
    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/subset/" ++ filename ++ ".flac", .{});
    defer file.close();

    const buffer = try allocator.alloc(u8, 8192);
    defer allocator.free(buffer);
    var reader = file.reader(buffer);

    return try zflac.decode(allocator, &reader.interface);
}

const PlayState = struct {
    file: *const zflac.DecodedFLAC,
    current_sample: usize,
    progress: std.Progress.Node,
    progress_node: std.Progress.Node,

    pub fn init(file: *const zflac.DecodedFLAC) PlayState {
        const parent_node = std.Progress.start(.{});
        return .{
            .file = file,
            .current_sample = 0,
            .progress = parent_node,
            .progress_node = parent_node.start("Playing", file.sample_count()),
        };
    }

    pub fn deinit(self: *PlayState) void {
        self.progress.end();
    }

    pub fn fill(self: *PlayState, output: *anyopaque, frame_count: u32) void {
        switch (self.file.samples) {
            .s8 => self._fill(i8, i16, output, frame_count),
            .s16 => self._fill(i16, i16, output, frame_count),
            .s32 => self._fill(i32, i32, output, frame_count),
        }
    }

    pub fn _fill(self: *PlayState, comptime SampleType: type, comptime OutputType: type, output: *anyopaque, frame_count: u32) void {
        var out: [*]OutputType = @ptrCast(@alignCast(output));
        switch (self.file.samples) {
            inline else => |samples| {
                for (0..self.file.channels * frame_count) |i| {
                    if (SampleType == i8 and OutputType == i16) {
                        out[i] = @as(i16, @intCast(samples[self.current_sample])) * 256;
                    } else {
                        out[i] = @intCast(samples[self.current_sample]);
                    }
                    self.current_sample += 1;
                    self.current_sample %= samples.len;
                    self.progress_node.completeOne();
                    if (self.current_sample == 0) {
                        self.progress_node.end();
                        self.progress_node = self.progress.start("Playing", self.file.sample_count());
                        std.log.info("Looping back...", .{});
                    }
                }
            },
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const r = try decode_standard_test(allocator, "23 - 8 bit per sample");
    defer r.deinit(allocator);

    std.debug.print("Decoded:\n", .{});
    std.debug.print("  Channel count: {d}\n", .{r.channels});
    std.debug.print("  Sample rate: {d}\n", .{r.sample_rate});
    std.debug.print("  Bits per samples: {d}\n", .{r.bits_per_sample});
    std.debug.print("  Sample count: {d}\n", .{r.sample_count()});

    zaudio.init(allocator);
    defer zaudio.deinit();

    var play_state: PlayState = .init(&r);
    defer play_state.deinit();

    var audio_device_config = zaudio.Device.Config.init(.playback);
    audio_device_config.sample_rate = r.sample_rate;
    audio_device_config.data_callback = audio_callback;
    audio_device_config.user_data = &play_state;
    audio_device_config.period_size_in_frames = 16;
    audio_device_config.playback.format = switch (r.samples) {
        .s8 => .signed16,
        .s16 => .signed16,
        .s32 => .signed32,
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
) callconv(.c) void {
    const state: *PlayState = @ptrCast(@alignCast(device.getUserData()));

    if (output) |out| {
        state.fill(out, frame_count);
    }
}
