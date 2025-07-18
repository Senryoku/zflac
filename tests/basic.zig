const std = @import("std");
const zflac = @import("zflac");

test "Example 1" {
    const Example = [_]u8{
        0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x0f, 0x00, 0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00,
        0x00, 0x01, 0x3e, 0x84, 0xb4, 0x18, 0x07, 0xdc, 0x69, 0x03, 0x07, 0x58,
        0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f, 0xff, 0xf8, 0x69, 0x18, 0x00, 0x00,
        0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa, 0x9a,
    };

    var file = std.io.fixedBufferStream(&Example);

    var r = try zflac.decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    try std.testing.expectEqual(2, r.channels);
    try std.testing.expectEqual(2, r.samples.s16.len);
    try std.testing.expectEqual(25588, r.samples.s16[0]);
    try std.testing.expectEqual(10416, r.samples.s16[1]);
}

test "Example 2" {
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

    var r = try zflac.decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    try std.testing.expectEqual(2, r.channels);
    try std.testing.expectEqual(2 * (16 + 3), r.samples.s16.len);
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
    }, r.samples.s16);
}

test "Example 3" {
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

    var r = try zflac.decode(std.testing.allocator, file.reader());
    defer r.deinit(std.testing.allocator);

    try std.testing.expectEqual(1, r.channels);
    try std.testing.expectEqual(24, r.samples.s8.len);
    try std.testing.expectEqualSlices(i8, &[_]i8{ 0, 79, 111, 78, 8, -61, -90, -68, -13, 42, 67, 53, 13, -27, -46, -38, -12, 14, 24, 19, 6, -4, -5, 0 }, r.samples.s8);
}
