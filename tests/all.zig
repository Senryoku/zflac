pub const basic = @import("basic.zig");
pub const std_subset = @import("std_subset.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
