const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library

    const lib_mod = b.addModule("zflac", .{
        .root_source_file = b.path("src/zflac.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Test

    const test_step = b.step("test", "Run unit tests");

    const basic_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/basic.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zflac", .module = lib_mod },
        },
    }) });
    const run_basic_tests = b.addRunArtifact(basic_tests);
    test_step.dependOn(&run_basic_tests.step);

    const std_subset_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/std_subset.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zflac", .module = lib_mod },
        },
    }) });
    const run_std_subset_tests = b.addRunArtifact(std_subset_tests);
    test_step.dependOn(&run_std_subset_tests.step);

    const std_uncommon_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/std_uncommon.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zflac", .module = lib_mod },
        },
    }) });
    const run_std_uncommon_tests = b.addRunArtifact(std_uncommon_tests);
    test_step.dependOn(&run_std_uncommon_tests.step);

    const std_faulty_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/std_faulty.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zflac", .module = lib_mod },
        },
    }) });
    const run_std_faulty_tests = b.addRunArtifact(std_faulty_tests);
    test_step.dependOn(&run_std_faulty_tests.step);

    // Examples (currently used for testing while developing)

    const maybe_zaudio = b.lazyDependency("zaudio", .{});
    if (maybe_zaudio) |zaudio| {
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

        example.root_module.addImport("zaudio", zaudio.module("root"));
        example.linkLibrary(zaudio.artifact("miniaudio"));

        b.installArtifact(example);

        const run_example = b.addRunArtifact(example);
        run_example.step.dependOn(b.getInstallStep());
        const run_step = b.step("run", "Run example");
        run_step.dependOn(&run_example.step);
    }
    const maybe_zbench = b.lazyDependency("zbench", .{});
    if (maybe_zbench) |zbench| {
        const benchmark_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/std_subset.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zflac", .module = lib_mod },
                .{ .name = "zbench", .module = zbench.module("zbench") },
            },
        });

        const benchmark = b.addExecutable(.{
            .name = "benchmark",
            .root_module = benchmark_module,
        });

        b.installArtifact(benchmark);

        const run_benchmark = b.addRunArtifact(benchmark);
        run_benchmark.step.dependOn(b.getInstallStep());
        const run_step = b.step("bench", "Run benchmark");
        run_step.dependOn(&run_benchmark.step);
    }
}
