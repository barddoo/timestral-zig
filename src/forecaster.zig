//! Block-autoregressive quantile forecaster, mirroring `timestral.forecaster`: forward-fill,
//! left-pad, decode block by block feeding back medians (targets), the real future (known
//! covariates) or a blank (past-only covariates), the advancing refine frame, the quantile
//! cap, categorical target encoding, and the clampsym test-time augmentation.
//!
//! Layout: series `b` of `B`, each with `V` target variates, `Cp` past-only and `Ck`
//! known-future covariate rows. Rows of one series are contiguous in that order.

const std = @import("std");
const model_mod = @import("model.zig");
const nn = @import("nn.zig");
const par = @import("par.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;
const Role = model_mod.Role;

pub const Options = struct {
    /// Context cap in time steps (most recent kept); 0 = the trained context length.
    max_context: usize = 0,
    /// Patches decoded per block, capped at the trained block size.
    per_shot_horizon_patches: usize = 32,
    /// Clamp to `[min - k * s, max + k * s]` of the observed context; 0 disables.
    quantile_real_cap_k: f32 = 1e4,
    /// Fed-back medians at or above this magnitude become 0.
    extreme_value_threshold: f32 = 1e30,
    symmetric_averaging: bool = true,
    clip_nonnegative: bool = true,
};

/// Covariates of a batch. Numeric arrays are `(B, C, len)` float32 with NaN for missing,
/// categorical ones `(B, C, len)` float64 category codes (NaN = missing). Past arrays have the
/// context length `T`; known arrays `T + horizon`.
pub const Covariates = struct {
    past: ?[]const f32 = null,
    n_past: usize = 0,
    known: ?[]const f32 = null,
    n_known: usize = 0,
    cat_past: ?[]const f64 = null,
    n_cat_past: usize = 0,
    cat_known: ?[]const f64 = null,
    n_cat_known: usize = 0,
};

const Rows = struct {
    past: ?[]const f32,
    n_past: usize,
    known: ?[]const f32,
    n_known: usize,
};

/// Forecast `horizon` steps after `inputs` `(B, V, T)` (zeros where unobserved) with `observed`
/// the same shape. Returns `(B, V, horizon, Q)` quantiles; the caller frees the slice.
pub fn forecast(
    allocator: Allocator,
    io: std.Io,
    model: *const Model,
    inputs: []const f32,
    observed: []const bool,
    b: usize,
    v: usize,
    t: usize,
    horizon: usize,
    cov: Covariates,
    opts: Options,
) ![]f32 {
    const cfgm = model.config;
    const q = cfgm.numQuantiles();
    var o = opts;
    if (o.max_context == 0) o.max_context = cfgm.context_length;
    o.per_shot_horizon_patches = @min(o.per_shot_horizon_patches, cfgm.block_size);

    const any_cov = cov.past != null or cov.known != null or cov.cat_past != null or cov.cat_known != null;
    if (any_cov and !cfgm.covariate_enabled) return error.CovariatesUnsupported;
    if ((cov.known != null or cov.cat_known != null) and !cfgm.supportsKnownCovariates()) return error.KnownCovariatesUnsupported;

    // Categorical codes are target-encoded against a univariate target, else used raw; the
    // encoded rows are appended after the numeric covariates.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var target: ?[]f64 = null;
    if ((cov.cat_past != null or cov.cat_known != null) and v == 1) {
        target = try encodingTarget(arena, inputs, observed, b, t, o.max_context);
    }
    const enc_past = if (cov.cat_past) |codes| try encodeCategorical(arena, codes, b, cov.n_cat_past, t, target, t) else null;
    const enc_known = if (cov.cat_known) |codes| try encodeCategorical(arena, codes, b, cov.n_cat_known, t + horizon, target, t) else null;
    const rows: Rows = .{
        .past = try catChannels(arena, cov.past, cov.n_past, enc_past, cov.n_cat_past, b, t),
        .n_past = cov.n_past + cov.n_cat_past,
        .known = try catChannels(arena, cov.known, cov.n_known, enc_known, cov.n_cat_known, b, t + horizon),
        .n_known = cov.n_known + cov.n_cat_known,
    };
    if (rows.n_past == 0 and rows.n_known == 0 and any_cov) return error.InvalidInput;

    var fwd: Pass = .{ .allocator = allocator, .io = io, .model = model, .inputs = inputs, .observed = observed, .b = b, .v = v, .t = t, .horizon = horizon, .rows = rows, .opts = o };
    if (o.symmetric_averaging) {
        const levels = cfgm.quantile_knots;
        for (levels, 0..) |lo, i| {
            if (@abs(lo + levels[levels.len - 1 - i] - 1.0) > 1e-9) return error.AsymmetricQuantiles;
        }
        // The mirrored pass forecasts -context with every covariate row negated. It is independent
        // of the forward pass, so the two run concurrently.
        const neg = try negated(arena, inputs);
        const mrows: Rows = .{
            .past = if (rows.past) |p| try negated(arena, p) else null,
            .n_past = rows.n_past,
            .known = if (rows.known) |k| try negated(arena, k) else null,
            .n_known = rows.n_known,
        };
        var mir = fwd;
        mir.inputs = neg;
        mir.rows = mrows;
        par.both(io, &fwd, Pass.run, &mir, Pass.run);
        const mirrored = mir.result catch |err| {
            if (fwd.result) |r| allocator.free(r) else |_| {}
            return err;
        };
        defer allocator.free(mirrored);
        const result = try fwd.result;
        errdefer allocator.free(result);
        for (0..b * v * horizon) |i| {
            const a = result[i * q ..][0..q];
            const m = mirrored[i * q ..][0..q];
            for (0..q) |j| a[j] = 0.5 * (a[j] + -m[q - 1 - j]);
        }
        return finish(result, inputs, observed, b, v, t, horizon, q, o);
    }
    fwd.run();
    return finish(try fwd.result, inputs, observed, b, v, t, horizon, q, o);
}

const Pass = struct {
    allocator: Allocator,
    io: std.Io,
    model: *const Model,
    inputs: []const f32,
    observed: []const bool,
    b: usize,
    v: usize,
    t: usize,
    horizon: usize,
    rows: Rows,
    opts: Options,
    result: anyerror![]f32 = error.NotRun,

    fn run(self: *Pass) void {
        self.result = forecastRaw(self.allocator, self.io, self.model, self.inputs, self.observed, self.b, self.v, self.t, self.horizon, self.rows, self.opts);
    }
};

/// Non-negative clipping, applied after the passes are combined.
fn finish(result: []f32, inputs: []const f32, observed: []const bool, b: usize, v: usize, t: usize, horizon: usize, q: usize, o: Options) []f32 {
    if (o.clip_nonnegative) {
        for (0..b * v) |r| {
            var any = false;
            var all_nonneg = true;
            for (0..t) |i| {
                if (observed[r * t + i] and std.math.isFinite(inputs[r * t + i])) {
                    any = true;
                    all_nonneg = all_nonneg and inputs[r * t + i] >= 0;
                }
            }
            if (any and all_nonneg) for (result[r * horizon * q ..][0 .. horizon * q]) |*x| {
                x.* = @max(x.*, 0);
            };
        }
    }
    return result;
}

fn negated(allocator: Allocator, x: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, x.len);
    for (out, x) |*a, bv| a.* = -bv;
    return out;
}

/// `(B, Ca + Cb, len)` from two `(B, C, len)` channel groups (numeric first), either optional.
fn catChannels(allocator: Allocator, a: ?[]const f32, ca: usize, bb: ?[]const f32, cb: usize, batch: usize, len: usize) !?[]const f32 {
    if (ca + cb == 0) return null;
    const out = try allocator.alloc(f32, batch * (ca + cb) * len);
    for (0..batch) |i| {
        if (a) |av| @memcpy(out[i * (ca + cb) * len ..][0 .. ca * len], av[i * ca * len ..][0 .. ca * len]);
        if (bb) |bv| @memcpy(out[(i * (ca + cb) + ca) * len ..][0 .. cb * len], bv[i * cb * len ..][0 .. cb * len]);
    }
    return out;
}

/// `(B, T)` target to encode against: the forward-filled target from its first observation on,
/// over the last `max_context` steps; NaN elsewhere.
fn encodingTarget(allocator: Allocator, inputs: []const f32, observed: []const bool, b: usize, t: usize, max_context: usize) ![]f64 {
    const out = try allocator.alloc(f64, b * t);
    for (0..b) |s| {
        var started = false;
        var last: f32 = 0;
        for (0..t) |i| {
            if (observed[s * t + i]) {
                started = true;
                last = inputs[s * t + i];
            }
            const in_window = t <= max_context or i >= t - max_context;
            out[s * t + i] = if (started and in_window) last else std.math.nan(f64);
        }
    }
    return out;
}

/// Target-encode `(B, C, len)` codes against `(B, T)` targets (NaN outside the encoding
/// window), or pass them through as float32 when there is no univariate target.
fn encodeCategorical(allocator: Allocator, codes: []const f64, b: usize, c: usize, len: usize, target: ?[]const f64, t: usize) ![]f32 {
    const out = try allocator.alloc(f32, b * c * len);
    for (0..b) |s| {
        for (0..c) |j| {
            const row = codes[(s * c + j) * len ..][0..len];
            const dst = out[(s * c + j) * len ..][0..len];
            const tgt = target orelse {
                for (dst, row) |*x, cv| x.* = @floatCast(cv);
                continue;
            };
            try targetEncode(allocator, tgt[s * t ..][0..t], row[0..t], if (len > t) row[t..] else null, dst[0..t], if (len > t) dst[t..] else null);
            // A missing code before the target starts is padding, not the missing category.
            for (0..t) |i| if (std.math.isNan(tgt[s * t + i]) and std.math.isNan(row[i])) {
                dst[i] = std.math.nan(f32);
            };
        }
    }
    return out;
}

/// Per-series smoothed mean-target encoding (`timestral.covariate_encoding.target_encode`,
/// smooth = 1): category `c` becomes `(mean + sum_c) / (1 + count_c)`; NaN is its own category,
/// and a category absent from the context maps to the series mean.
fn targetEncode(allocator: Allocator, target: []const f64, past: []const f64, future: ?[]const f64, out_past: []f32, out_future: ?[]f32) !void {
    const smooth: f64 = 1.0;
    var item_count: f64 = 0;
    var item_sum: f64 = 0;
    for (target) |x| if (std.math.isFinite(x)) {
        item_count += 1;
        item_sum += x;
    };
    const item_mean = if (item_count > 0) item_sum / item_count else 0.0;

    var cats: std.ArrayList(f64) = .empty;
    defer cats.deinit(allocator);
    for (past) |cv| if (std.math.isFinite(cv)) try cats.append(allocator, cv);
    std.mem.sort(f64, cats.items, {}, std.sort.asc(f64));
    var nc: usize = 0;
    for (cats.items, 0..) |cv, i| {
        if (i == 0 or cv != cats.items[nc - 1]) {
            cats.items[nc] = cv;
            nc += 1;
        }
    }
    const categories = cats.items[0..nc];

    const sums = try allocator.alloc(f64, nc + 1);
    defer allocator.free(sums);
    const counts = try allocator.alloc(f64, nc + 1);
    defer allocator.free(counts);
    @memset(sums, 0);
    @memset(counts, 0);
    for (past, target) |cv, x| {
        if (!std.math.isFinite(x)) continue;
        const slot = slotOf(categories, cv) orelse nc;
        sums[slot] += x;
        counts[slot] += 1;
    }
    for (past, out_past) |cv, *o| {
        const slot = slotOf(categories, cv) orelse nc;
        o.* = @floatCast((smooth * item_mean + sums[slot]) / (smooth + counts[slot]));
    }
    if (future) |fc| {
        for (fc, out_future.?) |cv, *o| {
            // NaN is the missing category; a code never seen in the context is unseen.
            const v: f64 = if (std.math.isNan(cv)) (smooth * item_mean + sums[nc]) / (smooth + counts[nc]) else if (slotOf(categories, cv)) |slot| (smooth * item_mean + sums[slot]) / (smooth + counts[slot]) else item_mean;
            o.* = @floatCast(v);
        }
    }
}

/// Index of `code` in the sorted `categories`; NaN and unknown codes give null.
fn slotOf(categories: []const f64, code: f64) ?usize {
    if (std.math.isNan(code)) return null;
    var lo: usize = 0;
    var hi: usize = categories.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (categories[mid] < code) lo = mid + 1 else hi = mid;
    }
    return if (lo < categories.len and categories[lo] == code) lo else null;
}

fn forecastRaw(
    allocator: Allocator,
    io: std.Io,
    model: *const Model,
    inputs: []const f32,
    observed: []const bool,
    b: usize,
    v: usize,
    t0: usize,
    horizon: usize,
    cov: Rows,
    o: Options,
) ![]f32 {
    const c = model.config;
    const ps: usize = c.patch_size;
    const q = c.numQuantiles();
    const vall = v + cov.n_past + cov.n_known;
    const nrows = b * vall;

    // Roles of one series' rows.
    const roles = try allocator.alloc(Role, vall);
    defer allocator.free(roles);
    for (roles, 0..) |*r, i| r.* = if (i < v) .target else if (i < v + cov.n_past) .past else .known;

    // Keep the last max_context steps, left-pad to a patch multiple.
    const keep = @min(t0, o.max_context);
    const t = (keep + ps - 1) / ps * ps;
    const rows = try allocator.alloc(f32, nrows * t);
    defer allocator.free(rows);
    const mask = try allocator.alloc(bool, nrows * t);
    defer allocator.free(mask);
    @memset(rows, 0);
    @memset(mask, false);
    const hpad = (horizon + ps - 1) / ps * ps;
    const kf = try allocator.alloc(f32, nrows * hpad);
    defer allocator.free(kf);
    const kfm = try allocator.alloc(bool, nrows * hpad);
    defer allocator.free(kfm);
    @memset(kf, 0);
    @memset(kfm, false);

    for (0..b) |s| {
        for (0..vall) |i| {
            const r = s * vall + i;
            var last: f32 = 0;
            var seen = false;
            for (0..t0) |j| {
                var val: f32 = undefined;
                var ob: bool = undefined;
                switch (roles[i]) {
                    .target => {
                        // Forward-fill the target; leading gaps stay 0.
                        const idx = (s * v + i) * t0 + j;
                        if (observed[idx]) {
                            last = inputs[idx];
                            seen = true;
                        }
                        val = if (seen) last else 0;
                        ob = observed[idx];
                    },
                    .past => {
                        const x = cov.past.?[(s * cov.n_past + (i - v)) * t0 + j];
                        ob = !std.math.isNan(x);
                        val = if (ob) x else 0;
                    },
                    .known => {
                        const x = cov.known.?[(s * cov.n_known + (i - v - cov.n_past)) * (t0 + horizon) + j];
                        ob = !std.math.isNan(x);
                        val = if (ob) x else 0;
                    },
                }
                if (j >= t0 - keep) {
                    const dst = r * t + (t - keep) + (j - (t0 - keep));
                    rows[dst] = val;
                    mask[dst] = ob;
                }
            }
            if (roles[i] == .known) {
                for (0..horizon) |j| {
                    const x = cov.known.?[(s * cov.n_known + (i - v - cov.n_past)) * (t0 + horizon) + t0 + j];
                    kfm[r * hpad + j] = !std.math.isNan(x);
                    kf[r * hpad + j] = if (kfm[r * hpad + j]) x else 0;
                }
            }
        }
    }

    const nt = b * v;
    const out = try allocator.alloc(f32, nt * horizon * q);
    errdefer allocator.free(out);
    const known_any = cov.n_known > 0;
    try blockDecode(io, allocator, model, out, rows, mask, if (known_any) kf else null, if (known_any) kfm else null, hpad, b, roles, t, horizon, o);

    if (o.quantile_real_cap_k > 0) {
        const n = t / ps;
        const loc = try allocator.alloc(f32, n);
        defer allocator.free(loc);
        const scale = try allocator.alloc(f32, n);
        defer allocator.free(scale);
        for (0..b) |s| {
            for (0..v) |i| {
                const r = s * vall + i;
                model.patchStats(io, loc, scale, rows[r * t ..][0..t], mask[r * t ..][0..t], 1, t);
                var mx: f32 = -std.math.inf(f32);
                var mn: f32 = std.math.inf(f32);
                for (0..t) |j| if (mask[r * t + j]) {
                    mx = @max(mx, rows[r * t + j]);
                    mn = @min(mn, rows[r * t + j]);
                };
                if (std.math.isInf(mx)) mx = 0;
                if (std.math.isInf(mn)) mn = 0;
                const a = scale[n - 1] * o.quantile_real_cap_k;
                for (out[(s * v + i) * horizon * q ..][0 .. horizon * q]) |*x| x.* = std.math.clamp(x.*, mn - a, mx + a);
            }
        }
    }
    for (out) |x| if (std.math.isNan(x)) return error.DegenerateInput;
    return out;
}

fn blockDecode(
    io: std.Io,
    allocator: Allocator,
    model: *const Model,
    out: []f32,
    rows: []const f32,
    mask: []const bool,
    kf: ?[]const f32,
    kfm: ?[]const bool,
    hpad: usize,
    groups: usize,
    roles: []const Role,
    t_in: usize,
    horizon: usize,
    o: Options,
) !void {
    const c = model.config;
    const ps: usize = c.patch_size;
    const q = c.numQuantiles();
    const med = c.medianIndex();
    const vall = roles.len;
    const nrows = groups * vall;
    var v: usize = 0; // target rows per series
    for (roles) |r| {
        if (r == .target) v += 1;
    }
    const nt = groups * v;

    var t = t_in;
    var ctx = try allocator.dupe(f32, rows);
    defer allocator.free(ctx);
    var cmask = try allocator.dupe(bool, mask);
    defer allocator.free(cmask);

    var remaining = (horizon + ps - 1) / ps;
    var done: usize = 0; // decoded steps so far
    while (remaining > 0) {
        const block = @min(o.per_shot_horizon_patches, remaining);
        const blen = block * ps;
        const qs = try allocator.alloc(f32, nt * blen * q);
        defer allocator.free(qs);

        var kfb: ?[]f32 = null;
        var kfmb: ?[]bool = null;
        if (kf) |kfv| {
            kfb = try allocator.alloc(f32, nrows * blen);
            kfmb = try allocator.alloc(bool, nrows * blen);
            for (0..nrows) |r| {
                @memcpy(kfb.?[r * blen ..][0..blen], kfv[r * hpad + done ..][0..blen]);
                @memcpy(kfmb.?[r * blen ..][0..blen], kfm.?[r * hpad + done ..][0..blen]);
            }
        }
        defer if (kfb) |x| allocator.free(x);
        defer if (kfmb) |x| allocator.free(x);

        try model.predictBlock(io, allocator, qs, .{
            .ctx = ctx,
            .obs = cmask,
            .t = t,
            .groups = groups,
            .roles = roles,
            .h = block,
            .known_future = kfb,
            .known_future_obs = kfmb,
        });
        if (block > 1) try refineBlock(io, allocator, model, qs, ctx, cmask, groups, vall, v, t, block);

        for (0..nt) |r| {
            for (0..blen) |s| {
                if (done + s >= horizon) break;
                @memcpy(out[(r * horizon + done + s) * q ..][0..q], qs[(r * blen + s) * q ..][0..q]);
            }
        }
        remaining -= block;
        done += blen;
        if (remaining == 0) break;

        // Feedback: targets carry their median, known covariates their real future, past-only
        // covariates a blank. Past max_context the oldest steps are dropped.
        const nt_len = t + blen;
        const keep_t = if (nt_len > o.max_context) @max(ps, o.max_context / ps * ps) else nt_len;
        const nctx = try allocator.alloc(f32, nrows * keep_t);
        const nmask = try allocator.alloc(bool, nrows * keep_t);
        for (0..groups) |g| {
            var ti: usize = 0;
            for (0..vall) |i| {
                const r = g * vall + i;
                const trow = g * v + ti;
                for (0..keep_t) |j| {
                    const src = nt_len - keep_t + j; // index in the extended row
                    if (src < t) {
                        nctx[r * keep_t + j] = ctx[r * t + src];
                        nmask[r * keep_t + j] = cmask[r * t + src];
                        continue;
                    }
                    const s = src - t;
                    switch (roles[i]) {
                        .target => {
                            var m = qs[(trow * blen + s) * q + med];
                            if (!std.math.isFinite(m) or @abs(m) >= o.extreme_value_threshold) m = 0;
                            nctx[r * keep_t + j] = m;
                            nmask[r * keep_t + j] = true;
                        },
                        .known => {
                            nctx[r * keep_t + j] = kfb.?[r * blen + s];
                            nmask[r * keep_t + j] = kfmb.?[r * blen + s];
                        },
                        .past => {
                            nctx[r * keep_t + j] = 0;
                            nmask[r * keep_t + j] = false;
                        },
                    }
                }
                if (roles[i] == .target) ti += 1;
            }
        }
        allocator.free(ctx);
        allocator.free(cmask);
        ctx = nctx;
        cmask = nmask;
        t = keep_t;
    }
}

const Welford = struct { w: f64, mean: f64, m2: f64 };

fn welfordCombine(a: Welford, b: Welford) Welford {
    const w = a.w + b.w;
    if (w == 0) return .{ .w = 0, .mean = 0, .m2 = 0 };
    const w_safe = @max(w, std.math.floatMin(f64));
    const delta = b.mean - a.mean;
    return .{
        .w = w,
        .mean = a.mean + delta * b.w / w_safe,
        .m2 = a.m2 + b.m2 + delta * delta * a.w * b.w / w_safe,
    };
}

/// Moves a decoded block `(T, h * patch_size, Q)` of the target rows from the anchor frame of
/// the last context patch to the advancing refine frame it was trained in: horizon patch `j` is
/// normalized with the statistics of the context plus the block's medians of patches `0..j-1`.
fn refineBlock(io: std.Io, allocator: Allocator, model: *const Model, qs: []f32, ctx: []const f32, mask: []const bool, groups: usize, vall: usize, v: usize, t: usize, h: usize) !void {
    const c = model.config;
    const ps: usize = c.patch_size;
    const q = c.numQuantiles();
    const med = c.medianIndex();
    const l = t / ps;

    const loc = try allocator.alloc(f32, l);
    defer allocator.free(loc);
    const scale = try allocator.alloc(f32, l);
    defer allocator.free(scale);

    const clip: f64 = @sqrt(@as(f64, std.math.floatMax(f32))) / 1e3;
    const sinh_med = try allocator.alloc(f64, h * ps);
    defer allocator.free(sinh_med);
    const loc_j = try allocator.alloc(f32, h);
    defer allocator.free(loc_j);
    const scale_j = try allocator.alloc(f32, h);
    defer allocator.free(scale_j);

    for (0..groups) |g| {
        for (0..v) |i| {
            const r = g * vall + i; // row in the packed context
            const row_ctx = ctx[r * t ..][0..t];
            const row_mask = mask[r * t ..][0..t];
            model.patchStats(io, loc, scale, row_ctx, row_mask, 1, t);
            const loc0 = loc[l - 1];
            const scale0 = scale[l - 1];
            const block = qs[(g * v + i) * h * ps * q ..][0 .. h * ps * q];
            for (0..h * ps) |s| sinh_med[s] = (@as(f64, block[s * q + med]) - @as(f64, loc0)) / @as(f64, scale0);

            // Welford state of the observed context, folded patch by patch.
            var state: Welford = .{ .w = 0, .mean = 0, .m2 = 0 };
            for (0..l) |p| {
                var cnt: f64 = 0;
                var sum: f64 = 0;
                for (0..ps) |k| if (row_mask[p * ps + k]) {
                    cnt += 1;
                    sum += row_ctx[p * ps + k];
                };
                var pm: Welford = .{ .w = cnt, .mean = 0, .m2 = 0 };
                if (cnt > 0) {
                    pm.mean = sum / @max(cnt, 1.0);
                    for (0..ps) |k| if (row_mask[p * ps + k]) {
                        const dv = @as(f64, row_ctx[p * ps + k]) - pm.mean;
                        pm.m2 += dv * dv;
                    };
                }
                state = welfordCombine(state, pm);
            }

            const offset = state.mean; // medians are clamped relative to the context mean
            for (0..h) |j| {
                const variance = @max(state.m2 / @max(state.w - 1.0, 1.0), 0.0);
                const sc = @max(@sqrt(variance), c.minimum_scale);
                loc_j[j] = @floatCast(state.mean);
                scale_j[j] = @floatCast(sc);
                var folded: [256]f64 = undefined; // patch_size is at most 256
                var bsum: f64 = 0;
                for (0..ps) |k| {
                    folded[k] = std.math.clamp(sinh_med[j * ps + k] * sc + state.mean - offset, -clip, clip) + offset;
                    bsum += folded[k];
                }
                const bmean = bsum / @as(f64, @floatFromInt(ps));
                var bm2: f64 = 0;
                for (0..ps) |k| bm2 += (folded[k] - bmean) * (folded[k] - bmean);
                state = welfordCombine(state, .{ .w = @floatFromInt(ps), .mean = bmean, .m2 = bm2 });
            }
            loc_j[0] = loc0;
            scale_j[0] = scale0;

            for (0..h * ps) |s| {
                const rowq = block[s * q ..][0..q];
                const lj = loc_j[s / ps];
                const sj = scale_j[s / ps];
                for (rowq) |*x| x.* = (x.* - loc0) / scale0 * sj + lj;
                nn.clampNonfinite(rowq);
            }
        }
    }
}
