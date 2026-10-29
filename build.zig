const std = @import("std");
const Translator = @import("translate_c").Translator;

const Blas = enum { accelerate, openblas, builtin };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_blas: Blas = if (target.result.os.tag == .macos) .accelerate else .builtin;
    const blas = b.option(Blas, "blas", "Matrix-product backend: accelerate (macOS), openblas, or builtin SIMD kernels") orelse default_blas;

    // The `c` module: the BLAS header translated to Zig, or an empty module for the builtin
    // kernels (which then need no C library at all).
    const c_mod = switch (blas) {
        .builtin => b.createModule(.{ .root_source_file = b.path("src/c_none.zig") }),
        .accelerate, .openblas => blk: {
            const header = b.addWriteFiles().add("c.h", switch (blas) {
                .accelerate => "#include <Accelerate/Accelerate.h>\n",
                else => "#include <cblas.h>\n",
            });
            const translate_c = b.dependency("translate_c", .{});
            const c: Translator = .init(translate_c, .{
                .c_source_file = header,
                .target = target,
                .optimize = optimize,
            });
            break :blk c.mod;
        },
    };

    const exe = b.addExecutable(.{
        .name = "timestral",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = blas != .builtin,
            .imports = &.{.{ .name = "c", .module = c_mod }},
        }),
    });
    switch (blas) {
        .accelerate => exe.root_module.linkFramework("Accelerate", .{}),
        .openblas => exe.root_module.linkSystemLibrary("openblas", .{}),
        .builtin => {},
    }
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the forecaster").dependOn(&run_cmd.step);
}
