const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/zflac.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib = b.addStaticLibrary(.{
        .name = "zflac",
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    const all_tests_module = b.createModule(.{
        .root_source_file = b.path("tests/all.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zflac", .module = lib_mod },
        },
    });
    const all_tests = b.addTest(.{
        .root_module = all_tests_module,
    });

    const run_lib_unit_tests = b.addRunArtifact(all_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // Examples (currently used for testing while developing)

    const example_module = b.createModule(.{
        .root_source_file = b.path("examples/example.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zflac", .module = lib_mod },
        },
    });
    const example = b.addExecutable(.{
        .name = "example",
        .root_module = example_module,
    });

    b.installArtifact(example);

    const run_example = b.addRunArtifact(example);
    run_example.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run example");
    run_step.dependOn(&run_example.step);
}
