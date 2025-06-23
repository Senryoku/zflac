const std = @import("std");
const builtin = @import("builtin");

const zflac = @import("zflac");
const zflac_ref = @import("zflac-ref");

const zbench = @import("zbench");

pub const std_options: std.Options = .{
    .log_level = .err,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        // .{ .scope = .zflac, .level = .info },
    },
};

const UTF8ConsoleOutput = struct {
    original: if (builtin.os.tag == .windows) c_uint else void,

    fn init() UTF8ConsoleOutput {
        if (builtin.os.tag == .windows) {
            const original = std.os.windows.kernel32.GetConsoleOutputCP();
            _ = std.os.windows.kernel32.SetConsoleOutputCP(65001);
            return .{ .original = original };
        }
        return .{ .original = {} };
    }

    fn deinit(self: UTF8ConsoleOutput) void {
        if (builtin.os.tag == .windows) {
            _ = std.os.windows.kernel32.SetConsoleOutputCP(self.original);
        }
    }
};

fn run_standard_test(comptime filename: []const u8, comptime impl: anytype) *const fn (std.mem.Allocator) void {
    return struct {
        fn run(allocator: std.mem.Allocator) void {
            const file = std.fs.cwd().openFile("test-files/ietf-wg-cellar/subset/" ++ filename ++ ".flac", .{}) catch |err| {
                std.debug.panic("Failed to open file: {s}", .{@errorName(err)});
            };
            defer file.close();

            var buffered_reader = std.io.bufferedReader(file.reader());
            const reader = buffered_reader.reader();

            var r = impl.decode(allocator, reader) catch |err| {
                std.debug.panic("Failed to decode FLAC: {s}", .{@errorName(err)});
            };
            defer r.deinit(allocator);
        }
    }.run;
}

pub fn main() !void {
    const cp_out = UTF8ConsoleOutput.init();
    defer cp_out.deinit();

    var bench = zbench.Benchmark.init(std.heap.page_allocator, .{});
    defer bench.deinit();
    inline for (&[_][]const u8{
        "01 - blocksize 4096",
        "02 - blocksize 4608",
        "03 - blocksize 16",
        "04 - blocksize 192",
        "05 - blocksize 254",
        "06 - blocksize 512",
        "07 - blocksize 725",
        "08 - blocksize 1000",
        "09 - blocksize 1937",
        "10 - blocksize 2304",
        "11 - partition order 8",
        "12 - qlp precision 15 bit",
        "13 - qlp precision 2 bit",
        "14 - wasted bits",
        "15 - only verbatim subframes",
        "16 - partition order 8 containing escaped partitions",
        "17 - all fixed orders",
        "18 - precision search",
        "19 - samplerate 35467Hz",
        "20 - samplerate 39kHz",
        "21 - samplerate 22050Hz",
        "22 - 12 bit per sample",
        "23 - 8 bit per sample",
        "24 - variable blocksize file created with flake revision 264",
        "25 - variable blocksize file created with flake revision 264, modified to create smaller blocks",
        "26 - variable blocksize file created with CUETools.Flake 2.1.6",
        "27 - old format variable blocksize file created with Flake 0.11",
        "28 - high resolution audio, default settings",
        "29 - high resolution audio, blocksize 16384",
        "30 - high resolution audio, blocksize 13456",
        "31 - high resolution audio, using only 32nd order predictors",
        "32 - high resolution audio, partition order 8 containing escaped partitions",
        "33 - samplerate 192kHz",
        "34 - samplerate 192kHz, using only 32nd order predictors",
        "35 - samplerate 134560Hz",
        "36 - samplerate 384kHz",
        "37 - 20 bit per sample",
        "38 - 3 channels (3.0)",
        "39 - 4 channels (4.0)",
        "40 - 5 channels (5.0)",
        "41 - 6 channels (5.1)",
        "42 - 7 channels (6.1)",
        "43 - 8 channels (7.1)",
        "44 - 8-channel surround, 192kHz, 24 bit, using only 32nd order predictors",
        "45 - no total number of samples set",
        "46 - no min-max framesize set",
        "47 - only STREAMINFO",
        "48 - Extremely large SEEKTABLE",
        "49 - Extremely large PADDING",
        "50 - Extremely large PICTURE",
        "51 - Extremely large VORBISCOMMENT",
        "52 - Extremely large APPLICATION",
        "53 - CUESHEET with very many indexes",
        "54 - 1000x repeating VORBISCOMMENT",
        "55 - file 48-53 combined",
        "56 - JPG PICTURE",
        "57 - PNG PICTURE",
        "58 - GIF PICTURE",
        "59 - AVIF PICTURE",
        "60 - mono audio",
        "61 - predictor overflow check, 16-bit",
        "62 - predictor overflow check, 20-bit",
        "63 - predictor overflow check, 24-bit",
        "64 - rice partitions with escape code zero",
    }) |filename| {
        try bench.add("(now) " ++ filename, run_standard_test(filename, zflac), .{});
        try bench.add("(ref) " ++ filename, run_standard_test(filename, zflac_ref), .{});
    }
    try bench.run(std.io.getStdOut().writer());
}
