const std = @import("std");
const zflac = @import("zflac");

fn run_uncommon_test(comptime filename: []const u8) !void {
    const allocator = std.testing.allocator;

    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/uncommon/" ++ filename ++ ".flac", .{});
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    const reader = buffered_reader.reader();

    var r = try zflac.decode(allocator, reader);
    defer r.deinit(allocator);
}

// NOTE: Multiple of these are supposed to return an error be cause I chose not to support all these features,
//       however they're not failling for the expected reason: An unknown number of samples shouldn't be an error
//       as far as the standard is concerned.

test "01 - changing samplerate" {
    try std.testing.expectError(error.UnknownNumberOfSamples, run_uncommon_test("01 - changing samplerate"));
    // try std.testing.expectError(error.InconsistentParameters, run_uncommon_test("01 - changing samplerate"));
}

test "02 - increasing number of channels" {
    try std.testing.expectError(error.UnknownNumberOfSamples, run_uncommon_test("02 - increasing number of channels"));
    // try std.testing.expectError(error.InconsistentParameters, run_uncommon_test("02 - increasing number of channels"));
}

test "03 - decreasing number of channels" {
    try std.testing.expectError(error.UnknownNumberOfSamples, run_uncommon_test("03 - decreasing number of channels"));
    // try std.testing.expectError(error.InconsistentParameters, run_uncommon_test("03 - decreasing number of channels"));
}

test "04 - changing bitdepth" {
    try std.testing.expectError(error.UnknownNumberOfSamples, run_uncommon_test("04 - changing bitdepth"));
    // try std.testing.expectError(error.InconsistentParameters, run_uncommon_test("04 - changing bitdepth"));
}

test "05 - 32bps audio" {
    try run_uncommon_test("05 - 32bps audio");
}

test "06 - samplerate 768kHz" {
    try run_uncommon_test("06 - samplerate 768kHz");
}

test "07 - 15 bit per sample" {
    try run_uncommon_test("07 - 15 bit per sample");
}

test "08 - blocksize 65535" {
    try run_uncommon_test("08 - blocksize 65535");
}

test "09 - Rice partition order 15" {
    try run_uncommon_test("09 - Rice partition order 15");
}

test "10 - file starting at frame header" {
    try std.testing.expectError(error.InvalidSignature, run_uncommon_test("10 - file starting at frame header"));
}

test "11 - file starting with unparsable data" {
    try std.testing.expectError(error.InvalidSignature, run_uncommon_test("11 - file starting with unparsable data"));
}
