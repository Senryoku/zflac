const std = @import("std");
const zflac = @import("zflac");

fn run_standard_test(comptime filename: []const u8) !void {
    const allocator = std.testing.allocator;

    const file = try std.fs.cwd().openFile("test-files/ietf-wg-cellar/subset/" ++ filename ++ ".flac", .{});
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    const reader = buffered_reader.reader();

    var r = try zflac.decode(allocator, reader);
    defer r.deinit(allocator);

    const expected_samples_file = try std.fs.cwd().openFile("tests/expected_samples/" ++ filename ++ ".raw", .{});
    defer expected_samples_file.close();
    const expected = try expected_samples_file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(expected);

    switch (r.sample_bit_size()) {
        8 => {
            // FIXME: This fails while the CRC matches and the playback sounds correct. Maybe I'm not supposed to output signed values for 8bits?
            // try std.testing.expectEqualSlices(i8, @as([*]const i8, @alignCast(@ptrCast(expected)))[0..expected.len], try r.samples(i8));
            var expected_i8 = try allocator.alloc(i8, expected.len);
            defer allocator.free(expected_i8);
            for (0..expected_i8.len) |i| {
                expected_i8[i] = @bitCast(expected[i] -% 128);
            }
            try std.testing.expectEqualSlices(i8, expected_i8, try r.samples(i8));
        },
        16 => try std.testing.expectEqualSlices(i16, @as([*]const i16, @alignCast(@ptrCast(expected)))[0 .. expected.len / 2], try r.samples(i16)),
        24 => try std.testing.expectEqualSlices(i32, @as([*]const i32, @alignCast(@ptrCast(expected)))[0 .. expected.len / 4], try r.samples(i32)),
        32 => try std.testing.expectEqualSlices(i32, @as([*]const i32, @alignCast(@ptrCast(expected)))[0 .. expected.len / 4], try r.samples(i32)),
        else => unreachable,
    }
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

test "35 - samplerate 134560Hz" {
    try run_standard_test("35 - samplerate 134560Hz");
}

test "36 - samplerate 384kHz" {
    try run_standard_test("36 - samplerate 384kHz");
}

test "37 - 20 bit per sample" {
    try run_standard_test("37 - 20 bit per sample");
}

test "38 - 3 channels (3.0)" {
    try run_standard_test("38 - 3 channels (3.0)");
}

test "39 - 4 channels (4.0)" {
    try run_standard_test("39 - 4 channels (4.0)");
}

test "40 - 5 channels (5.0)" {
    try run_standard_test("40 - 5 channels (5.0)");
}

test "41 - 6 channels (5.1)" {
    try run_standard_test("41 - 6 channels (5.1)");
}

test "42 - 7 channels (6.1)" {
    try run_standard_test("42 - 7 channels (6.1)");
}

test "43 - 8 channels (7.1)" {
    try run_standard_test("43 - 8 channels (7.1)");
}

test "44 - 8-channel surround, 192kHz, 24 bit, using only 32nd order predictors" {
    try run_standard_test("44 - 8-channel surround, 192kHz, 24 bit, using only 32nd order predictors");
}

test "45 - no total number of samples set" {
    try run_standard_test("45 - no total number of samples set");
}

test "46 - no min-max framesize set" {
    try run_standard_test("46 - no min-max framesize set");
}

test "47 - only STREAMINFO" {
    try run_standard_test("47 - only STREAMINFO");
}

test "48 - Extremely large SEEKTABLE" {
    try run_standard_test("48 - Extremely large SEEKTABLE");
}

test "49 - Extremely large PADDING" {
    try run_standard_test("49 - Extremely large PADDING");
}

test "50 - Extremely large PICTURE" {
    try run_standard_test("50 - Extremely large PICTURE");
}

test "51 - Extremely large VORBISCOMMENT" {
    try run_standard_test("51 - Extremely large VORBISCOMMENT");
}

test "52 - Extremely large APPLICATION" {
    try run_standard_test("52 - Extremely large APPLICATION");
}

test "53 - CUESHEET with very many indexes" {
    try run_standard_test("53 - CUESHEET with very many indexes");
}

test "54 - 1000x repeating VORBISCOMMENT" {
    try run_standard_test("54 - 1000x repeating VORBISCOMMENT");
}

test "55 - file 48-53 combined" {
    try run_standard_test("55 - file 48-53 combined");
}

test "56 - JPG PICTURE" {
    try run_standard_test("56 - JPG PICTURE");
}

test "57 - PNG PICTURE" {
    try run_standard_test("57 - PNG PICTURE");
}

test "58 - GIF PICTURE" {
    try run_standard_test("58 - GIF PICTURE");
}

test "59 - AVIF PICTURE" {
    try run_standard_test("59 - AVIF PICTURE");
}

test "60 - mono audio" {
    try run_standard_test("60 - mono audio");
}

test "61 - predictor overflow check, 16-bit" {
    try run_standard_test("61 - predictor overflow check, 16-bit");
}

test "62 - predictor overflow check, 20-bit" {
    try run_standard_test("62 - predictor overflow check, 20-bit");
}

test "63 - predictor overflow check, 24-bit" {
    try run_standard_test("63 - predictor overflow check, 24-bit");
}

test "64 - rice partitions with escape code zero" {
    try run_standard_test("64 - rice partitions with escape code zero");
}
