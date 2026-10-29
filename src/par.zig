//! Fork-join parallelism over `std.Io.Threaded`.
//!
//! `forRange` splits `0..n` into contiguous chunks and runs them on the pool, the first on the
//! calling thread. When every pool thread is busy (for example under a nested call) `Io.async`
//! runs the task inline, so nesting never deadlocks.

const std = @import("std");

pub var io: ?std.Io = null;
/// Number of threads work is split across, including the calling thread.
pub var threads: usize = 1;

pub fn forRange(n: usize, ctx: anytype, comptime f: anytype) void {
    forRangeMin(n, 1, ctx, f);
}

/// As `forRange`, with every chunk at least `min_chunk` items.
pub fn forRangeMin(n: usize, min_chunk: usize, ctx: anytype, comptime f: anytype) void {
    const pool = io orelse return f(ctx, 0, n);
    const chunks = @min(threads, (n + min_chunk - 1) / @max(min_chunk, 1));
    if (chunks <= 1) return f(ctx, 0, n);
    const size = (n + chunks - 1) / chunks;
    var group: std.Io.Group = .init;
    var start = size;
    while (start < n) : (start += size) group.async(pool, f, .{ ctx, start, @min(start + size, n) });
    f(ctx, 0, @min(size, n));
    group.await(pool) catch {};
}

/// Run two independent tasks, the second possibly on another thread.
pub fn both(ctx_a: anytype, comptime fa: anytype, ctx_b: anytype, comptime fb: anytype) void {
    const pool = io orelse {
        fa(ctx_a);
        fb(ctx_b);
        return;
    };
    var group: std.Io.Group = .init;
    group.async(pool, fb, .{ctx_b});
    fa(ctx_a);
    group.await(pool) catch {};
}
