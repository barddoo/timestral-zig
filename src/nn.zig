//! Dense tensor kernels on flat float32 slices. Matrix products go to Accelerate's `cblas_sgemm`.

const std = @import("std");
const c = @import("c");
const par = @import("par.zig");

/// Whether matrix products go to a C BLAS (Accelerate, OpenBLAS); otherwise the builtin SIMD kernels run.
pub const has_blas = @hasDecl(c, "cblas_sgemm");

/// `C[m, n] = alpha * A[m, k] @ B^T` with B stored `[n, k]`, explicit row strides.
pub fn gemmNT(cc: []f32, ldc: usize, a: []const f32, lda: usize, b: []const f32, ldb: usize, m: usize, n: usize, k: usize, alpha: f32) void {
    if (comptime has_blas) {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans, @intCast(m), @intCast(n), @intCast(k), alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), 0.0, cc.ptr, @intCast(ldc));
        return;
    }
    const kv = k / lanes * lanes;
    for (0..m) |i| {
        const ar = a[i * lda ..];
        var j: usize = 0;
        // Four B rows at a time so every load of A is reused four times.
        while (j + 4 <= n) : (j += 4) {
            var acc: [4]Vec = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            var p: usize = 0;
            while (p < kv) : (p += lanes) {
                const av: Vec = ar[p..][0..lanes].*;
                inline for (0..4) |u| {
                    const bv: Vec = b[(j + u) * ldb + p ..][0..lanes].*;
                    acc[u] += av * bv;
                }
            }
            inline for (0..4) |u| {
                var sum = @reduce(.Add, acc[u]);
                var q = kv;
                while (q < k) : (q += 1) sum += ar[q] * b[(j + u) * ldb + q];
                cc[i * ldc + j + u] = alpha * sum;
            }
        }
        while (j < n) : (j += 1) {
            var acc: Vec = @splat(0);
            var p: usize = 0;
            while (p < kv) : (p += lanes) {
                const av: Vec = ar[p..][0..lanes].*;
                const bv: Vec = b[j * ldb + p ..][0..lanes].*;
                acc += av * bv;
            }
            var sum = @reduce(.Add, acc);
            var q = kv;
            while (q < k) : (q += 1) sum += ar[q] * b[j * ldb + q];
            cc[i * ldc + j] = alpha * sum;
        }
    }
}

/// `C[m, n] = A[m, k] @ B[k, n]` with explicit row strides.
pub fn gemmNN(cc: []f32, ldc: usize, a: []const f32, lda: usize, b: []const f32, ldb: usize, m: usize, n: usize, k: usize) void {
    if (comptime has_blas) {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans, @intCast(m), @intCast(n), @intCast(k), 1.0, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), 0.0, cc.ptr, @intCast(ldc));
        return;
    }
    const nv = n / lanes * lanes;
    for (0..m) |i| {
        const crow = cc[i * ldc ..][0..n];
        @memset(crow, 0);
        for (0..k) |p| {
            const av = a[i * lda + p];
            if (av == 0) continue;
            const avv: Vec = @splat(av);
            const brow = b[p * ldb ..][0..n];
            var j: usize = 0;
            while (j < nv) : (j += lanes) {
                const cv: Vec = crow[j..][0..lanes].*;
                const bv: Vec = brow[j..][0..lanes].*;
                crow[j..][0..lanes].* = cv + avv * bv;
            }
            while (j < n) : (j += 1) crow[j] += av * brow[j];
        }
    }
}

const LinearCtx = struct {
    y: []f32,
    x: []const f32,
    rows: usize,
    w: []const f32,
    out: usize,
    in: usize,
    bias: ?[]const f32,
};

/// Output columns `[j0, j1)` of a linear layer; the weight rows of those columns are contiguous.
fn linearCols(ctx: *const LinearCtx, j0: usize, j1: usize) void {
    gemmNT(ctx.y[j0..], ctx.out, ctx.x, ctx.in, ctx.w[j0 * ctx.in ..], ctx.in, ctx.rows, j1 - j0, ctx.in, 1.0);
    if (ctx.bias) |b| {
        for (0..ctx.rows) |r| {
            for (ctx.y[r * ctx.out + j0 ..][0 .. j1 - j0], b[j0..j1]) |*v, bv| v.* += bv;
        }
    }
}

const col_block = 16;

fn linearBlocks(ctx: *const LinearCtx, b0: usize, b1: usize) void {
    const nblocks = (ctx.out + col_block - 1) / col_block;
    linearCols(ctx, b0 * col_block, if (b1 == nblocks) ctx.out else b1 * col_block);
}

/// `y[rows, out] = x[rows, in] @ w[out, in]^T`, plus `bias[out]` when given. Large products are
/// split by output columns across the thread pool.
pub fn linear(y: []f32, x: []const f32, rows: usize, w: []const f32, out: usize, in: usize, bias: ?[]const f32) void {
    std.debug.assert(y.len >= rows * out and x.len >= rows * in and w.len == out * in);
    const ctx: LinearCtx = .{ .y = y, .x = x, .rows = rows, .w = w, .out = out, .in = in, .bias = bias };
    if (par.threads > 1 and rows * out * in >= (1 << 20) and out >= 2 * col_block) {
        par.forRange((out + col_block - 1) / col_block, &ctx, linearBlocks);
    } else {
        linearCols(&ctx, 0, out);
    }
}

/// SIMD lane count for f32 on the build target.
pub const lanes = std.simd.suggestVectorLength(f32) orelse 4;
const Vec = @Vector(lanes, f32);
const IVec = @Vector(lanes, i32);

/// `exp` on a vector: range reduction `x = n ln2 + r` and a degree-6 polynomial for `e^r`
/// (Cephes `expf`, about 1 ulp). The input is clamped to the finite float32 range of `exp`.
inline fn expVec(x_in: Vec) Vec {
    const x = @min(@max(x_in, @as(Vec, @splat(-87.33654))), @as(Vec, @splat(88.72284)));
    const fx = @round(x * @as(Vec, @splat(1.44269504088896341)));
    var r = x - fx * @as(Vec, @splat(0.693359375));
    r = r - fx * @as(Vec, @splat(-2.12194440e-4));
    var y: Vec = @splat(1.9875691500e-4);
    y = y * r + @as(Vec, @splat(1.3981999507e-3));
    y = y * r + @as(Vec, @splat(8.3334519073e-3));
    y = y * r + @as(Vec, @splat(4.1665795894e-2));
    y = y * r + @as(Vec, @splat(1.6666665459e-1));
    y = y * r + @as(Vec, @splat(5.0000001201e-1));
    y = y * r * r + r + @as(Vec, @splat(1.0));
    const n: IVec = @intFromFloat(fx);
    const pow2: Vec = @bitCast((n + @as(IVec, @splat(127))) << @as(@Vector(lanes, u5), @splat(23)));
    return y * pow2;
}

const NormCtx = struct { out: []f32, x: []const f32, dim: usize, scale: []const f32, eps: f32 };

fn rmsNormRows(ctx: *const NormCtx, r0: usize, r1: usize) void {
    rmsNormSerial(ctx.out, ctx.x, r0, r1, ctx.dim, ctx.scale, ctx.eps);
}

pub fn rmsNorm(out: []f32, x: []const f32, rows: usize, dim: usize, scale: []const f32, eps: f32) void {
    const ctx: NormCtx = .{ .out = out, .x = x, .dim = dim, .scale = scale, .eps = eps };
    par.forRangeMin(rows, 64, &ctx, rmsNormRows);
}

fn rmsNormSerial(out: []f32, x: []const f32, r0: usize, r1: usize, dim: usize, scale: []const f32, eps: f32) void {
    for (r0..r1) |r| {
        const xr = x[r * dim ..][0..dim];
        var acc: Vec = @splat(0);
        var i: usize = 0;
        while (i + lanes <= dim) : (i += lanes) {
            const v: Vec = xr[i..][0..lanes].*;
            acc += v * v;
        }
        var ss: f32 = @reduce(.Add, acc);
        while (i < dim) : (i += 1) ss += xr[i] * xr[i];
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dim)) + eps);
        const o = out[r * dim ..][0..dim];
        const invv: Vec = @splat(inv);
        i = 0;
        while (i + lanes <= dim) : (i += lanes) {
            const v: Vec = xr[i..][0..lanes].*;
            const sc: Vec = scale[i..][0..lanes].*;
            o[i..][0..lanes].* = v * invv * sc;
        }
        while (i < dim) : (i += 1) o[i] = xr[i] * inv * scale[i];
    }
}

/// `dst[i] = gate[i] * silu(up[i])`.
pub fn mulSilu(dst: []f32, gate: []const f32, up: []const f32) void {
    const one: Vec = @splat(1.0);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        const u: Vec = up[i..][0..lanes].*;
        const g: Vec = gate[i..][0..lanes].*;
        dst[i..][0..lanes].* = g * (u / (one + expVec(-u)));
    }
    while (i < dst.len) : (i += 1) dst[i] = gate[i] * silu(up[i]);
}

/// `dst[i] = x[i] * sigmoid(g[i])`.
pub fn mulSigmoid(dst: []f32, x: []const f32, g: []const f32) void {
    const one: Vec = @splat(1.0);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        const xv: Vec = x[i..][0..lanes].*;
        const gv: Vec = g[i..][0..lanes].*;
        dst[i..][0..lanes].* = xv / (one + expVec(-gv));
    }
    while (i < dst.len) : (i += 1) dst[i] = x[i] * sigmoid(g[i]);
}

/// Softmax in place; entries equal to -inf get exactly zero weight.
pub fn softmax(row: []f32) void {
    const ninf = -std.math.inf(f32);
    var mx: f32 = ninf;
    for (row) |v| mx = @max(mx, v);
    const mxv: Vec = @splat(mx);
    const zero: Vec = @splat(0);
    const ninfv: Vec = @splat(ninf);
    var sumv: Vec = @splat(0);
    var i: usize = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        const v: Vec = row[i..][0..lanes].*;
        const e = @select(f32, v == ninfv, zero, expVec(v - mxv));
        row[i..][0..lanes].* = e;
        sumv += e;
    }
    var sum: f32 = @reduce(.Add, sumv);
    while (i < row.len) : (i += 1) {
        row[i] = if (row[i] == ninf) 0 else @exp(row[i] - mx);
        sum += row[i];
    }
    const inv: Vec = @splat(1.0 / sum);
    const invs = 1.0 / sum;
    i = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        const v: Vec = row[i..][0..lanes].*;
        row[i..][0..lanes].* = v * inv;
    }
    while (i < row.len) : (i += 1) row[i] *= invs;
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// `torch.nn.functional.softplus` (beta=1, threshold=20).
pub inline fn softplus(x: f32) f32 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

/// Maps +/-inf to the nearest finite value of the vector; an all non-finite vector becomes 0.
/// NaN is left untouched.
pub fn clampNonfinite(v: []f32) void {
    var hi: f32 = -std.math.inf(f32);
    var lo: f32 = std.math.inf(f32);
    var any_inf = false;
    for (v) |x| {
        if (std.math.isFinite(x)) {
            hi = @max(hi, x);
            lo = @min(lo, x);
        } else if (std.math.isInf(x)) any_inf = true;
    }
    if (!any_inf) return;
    for (v) |*x| {
        if (x.* == std.math.inf(f32)) x.* = hi;
        if (x.* == -std.math.inf(f32)) x.* = lo;
        if (std.math.isInf(x.*)) x.* = 0;
    }
}
