//! File and process helpers over `std.Io`. `io` is set once by `main`.

const std = @import("std");

pub var io: std.Io = undefined;

pub fn eprint(msg: []const u8) void {
    std.Io.File.stderr().writeStreamingAll(io, msg) catch {};
}

pub fn eprintf(comptime fmt: []const u8, args: anytype) void {
    var buf: [1024]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    eprint(s);
}

pub fn writeStdout(bytes: []const u8) !void {
    try std.Io.File.stdout().writeStreamingAll(io, bytes);
}

/// The whole file at `path`; the caller frees it.
pub fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

pub fn exists(path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// A read-only memory mapping of a whole file.
pub const Mapped = struct {
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    bytes: []const u8,

    pub fn open(path: []const u8) !Mapped {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        const map = try std.Io.File.MemoryMap.create(io, file, .{
            .len = len,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        });
        return .{ .file = file, .map = map, .bytes = map.memory[0..len] };
    }

    pub fn close(self: *Mapped) void {
        self.map.destroy(io);
        self.file.close(io);
    }
};

/// Seconds on a monotonic-enough wall clock, for timing.
pub fn now() f64 {
    const ns = std.Io.Timestamp.now(io, .real).nanoseconds;
    return @as(f64, @floatFromInt(ns)) / 1e9;
}
