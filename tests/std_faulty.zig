const std = @import("std");
const zflac = @import("zflac");

fn run_faulty_test(comptime filename: []const u8) !void {
    const allocator = std.testing.allocator;

    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/faulty/" ++ filename ++ ".flac", .{});
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    const reader = buffered_reader.reader();

    var r = try zflac.decode(allocator, reader);
    defer r.deinit(allocator);
}

test "01 - wrong max blocksize" {
    // NOTE: Max blocksize from streaminfo metadata block is currently ignored.
    try run_faulty_test("01 - wrong max blocksize");
}

test "02 - wrong maximum framesize" {
    // NOTE: Max framesize from streaminfo metadata block is currently ignored.
    try run_faulty_test("02 - wrong maximum framesize");
}

test "03 - wrong bit depth" {
    try std.testing.expectError(error.InvalidChecksum, run_faulty_test("03 - wrong bit depth"));
}

test "04 - wrong number of channels" {
    try std.testing.expectError(error.InconsistentParameters, run_faulty_test("04 - wrong number of channels"));
}

test "05 - wrong total number of samples" {
    try run_faulty_test("05 - wrong total number of samples");
}

test "06 - missing streaminfo metadata block" {
    try std.testing.expectError(error.MissingStreaminfo, run_faulty_test("06 - missing streaminfo metadata block"));
}

test "07 - other metadata blocks preceding streaminfo metadata block" {
    try run_faulty_test("07 - other metadata blocks preceding streaminfo metadata block");
}

test "08 - blocksize 65536" {
    try std.testing.expectError(error.InvalidFrameHeader, run_faulty_test("08 - blocksize 65536"));
}

test "09 - blocksize 1" {
    try std.testing.expectError(error.InvalidFrameHeader, run_faulty_test("09 - blocksize 1"));
}

test "10 - invalid vorbis comment metadata block" {
    try run_faulty_test("10 - invalid vorbis comment metadata block");
}

test "11 - incorrect metadata block length" {
    try std.testing.expectError(error.InvalidMetadataHeader, run_faulty_test("11 - incorrect metadata block length"));
}
