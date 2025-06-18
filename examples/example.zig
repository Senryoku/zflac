const std = @import("std");

const zflac = @import("zflac");

pub const log_level: std.log.Level = .debug;

fn run_standard_test(comptime filename: []const u8) !void {
    const allocator = std.heap.page_allocator;

    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/subset/" ++ filename ++ ".flac", .{});
    defer file.close();

    var r = try zflac.decode(allocator, file.reader());
    defer r.deinit(allocator);
}

pub fn main() !void {
    try run_standard_test("23 - 8 bit per sample");
}
