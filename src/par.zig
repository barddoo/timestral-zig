//! Fork-join parallelism over `std.Io.Threaded`.
//!
//! `forRange` splits `0..n` into contiguous chunks and runs them on the pool, the first on the
//! calling thread. When every pool thread is busy (for example under a nested call) `Io.async`
//! runs the task inline, so nesting never deadlocks.

const std = @import("std");

/// Number of threads work is split across, including the calling thread.
pub var threads: usize = 1;

pub fn forRange(io: std.Io, n: usize, ctx: anytype, comptime f: anytype) void {
    forRangeMin(io, n, 1, ctx, f);
}

/// As `forRange`, with every chunk at least `min_chunk` items.
pub fn forRangeMin(io: std.Io, n: usize, min_chunk: usize, ctx: anytype, comptime f: anytype) void {
    const chunks = @min(threads, (n + min_chunk - 1) / @max(min_chunk, 1));
    if (chunks <= 1) return f(ctx, 0, n);
    const size = (n + chunks - 1) / chunks;
    var group: std.Io.Group = .init;
    var start = size;
    while (start < n) : (start += size) group.async(io, f, .{ ctx, start, @min(start + size, n) });
    f(ctx, 0, @min(size, n));
    group.await(io) catch {};
}

/// Run two independent tasks, the second possibly on another thread.
pub fn both(io: std.Io, ctx_a: anytype, comptime fa: anytype, ctx_b: anytype, comptime fb: anytype) void {
    var group: std.Io.Group = .init;
    group.async(io, fb, .{ctx_b});
    fa(ctx_a);
    group.await(io) catch {};
}
