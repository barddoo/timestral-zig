//! The Timestral network (context-only path): causal per-patch normalization, patch
//! embedding, a time/variate transformer over `[Enc ; Dec]` tokens, and the monotone
//! quantile head. Mirrors `timestral.model`, `timestral.layers` and `timestral.masks`.
//!
//! Tensors use the packed layout: `R` rows (the target variates and covariates of every
//! series), `N` patches, `D` channels, stored `[R][N][D]`. Rows of one series are contiguous.

const std = @import("std");
const cfg = @import("config.zig");
const nn = @import("nn.zig");
const st = @import("safetensors.zig");
const par = @import("par.zig");

const Allocator = std.mem.Allocator;

const Linear = struct {
    w: []const f32,
    b: ?[]const f32 = null,
};

const Attn = struct {
    is_time: bool,
    norm: []const f32,
    qkv: []const f32,
    out: []const f32,
    gate: ?[]const f32,
};

const MlpAct = struct {
    h: []const f32,
    a: []f32,
    hid: usize,

    /// `gate * silu(up)` for tokens `[first, last)`; the fused projection holds `[gate | up]` per token.
    fn run(self: *const MlpAct, first: usize, last: usize) void {
        for (first..last) |i| {
            const hv = self.h[i * 2 * self.hid ..][0 .. 2 * self.hid];
            nn.mulSilu(self.a[i * self.hid ..][0..self.hid], hv[0..self.hid], hv[self.hid..]);
        }
    }
};

const Layer = struct {
    attns: [2]Attn,
    n_attn: usize,
    mlp_norm: []const f32,
    w12: []const f32,
    w3: []const f32,
};

const ResBlock = struct {
    hidden: Linear,
    output: Linear,
    residual: Linear,
    in: usize,
    hid: usize,
    out: usize,

    fn load(w: *st.Weights, buf: []u8, prefix: []const u8, in: usize, hid: usize, out: usize, bias: bool) !ResBlock {
        return .{
            .hidden = try loadLinear(w, buf, prefix, "hidden_layer", hid, in, bias),
            .output = try loadLinear(w, buf, prefix, "output_layer", out, hid, bias),
            .residual = try loadLinear(w, buf, prefix, "residual_layer", out, in, bias),
            .in = in,
            .hid = hid,
            .out = out,
        };
    }

    /// `output(silu(hidden(x))) + residual(x)` for `rows` input vectors.
    fn forward(self: ResBlock, allocator: Allocator, y: []f32, x: []const f32, rows: usize) !void {
        const h = try allocator.alloc(f32, rows * self.hid);
        defer allocator.free(h);
        nn.linear(h, x, rows, self.hidden.w, self.hid, self.in, self.hidden.b);
        for (h) |*v| v.* = nn.silu(v.*);
        nn.linear(y, h, rows, self.output.w, self.out, self.hid, self.output.b);
        const r = try allocator.alloc(f32, rows * self.out);
        defer allocator.free(r);
        nn.linear(r, x, rows, self.residual.w, self.out, self.in, self.residual.b);
        for (y[0 .. rows * self.out], r) |*a, b| a.* += b;
    }
};

fn tensorName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

fn loadLinear(w: *st.Weights, buf: []u8, prefix: []const u8, name: []const u8, out: usize, in: usize, bias: bool) !Linear {
    const wn = tensorName(buf, "{s}.{s}.weight", .{ prefix, name });
    const weight = try w.get(wn, out * in);
    var bv: ?[]const f32 = null;
    if (bias) {
        const bn = tensorName(buf, "{s}.{s}.bias", .{ prefix, name });
        bv = try w.get(bn, out);
    }
    return .{ .w = weight, .b = bv };
}

/// What a row of the packed layout is. Covariates condition the targets but never read them;
/// only known-future covariates see the forecast horizon.
pub const Role = enum { target, past, known };

/// One decode block's input. Rows are grouped by series: `groups` series of `roles.len` rows
/// each, in the order given by `roles` (the same for every series).
pub const BlockInput = struct {
    ctx: []const f32, // (R, T)
    obs: []const bool, // (R, T)
    t: usize,
    groups: usize,
    roles: []const Role,
    h: usize, // patches to forecast
    /// `(R, h * patch_size)` known-future covariate values and their observation mask.
    known_future: ?[]const f32 = null,
    known_future_obs: ?[]const bool = null,
};

const Geo = struct {
    rows: usize,
    groups: usize,
    vall: usize, // rows per group
    ne: usize, // encoder tokens per row
    l: usize, // context patches
    h: usize, // decoder tokens per row
    roles: []const Role,
    cov: bool, // covariate window path (firewalled variate attention, bidirectional known rows)
    key_valid: []const bool, // (R, ne)
    first_obs: []const usize, // per group

    fn s(self: Geo) usize {
        return self.ne + self.h;
    }
    fn role(self: Geo, r: usize) Role {
        return self.roles[r % self.vall];
    }
};

pub const Model = struct {
    config: cfg.Config,
    patch_proj: ResBlock,
    forecast_query: []const f32,
    layers: []Layer,
    out_norm: []const f32,
    head: ResBlock,
    inv_freq: []f32, // head_dim / 2 entries; zero past rope_dim / 2 (identity rotation)

    pub fn load(allocator: Allocator, config: cfg.Config, w: *st.Weights) !Model {
        const d: usize = config.d_model;
        const ps: usize = config.patch_size;
        const hid: usize = config.mlp_hidden_dim;
        var buf: [256]u8 = undefined;

        const patch_proj = try ResBlock.load(w, &buf, "representation.patch_proj", 2 * ps, 4 * d, d, true);
        const forecast_query = try w.get("representation.forecast_query", d);
        const head = try ResBlock.load(w, &buf, "head.projection", d, 4 * d, ps * config.numQuantiles(), false);
        const out_norm = try w.get("transformer.out_norm.scale", d);

        const layers = try allocator.alloc(Layer, config.block_schedule.len);
        errdefer allocator.free(layers);
        for (config.block_schedule, 0..) |kind, li| {
            const axes = try cfg.layerAxes(kind);
            var layer: Layer = undefined;
            layer.n_attn = axes.len;
            for (0..axes.len) |ai| {
                const is_time = axes.is_time[ai];
                const gate_kind = if (is_time) config.time_attn_gate else config.variate_attn_gate;
                var a: Attn = undefined;
                a.is_time = is_time;
                a.norm = try w.get(tensorName(&buf, "transformer.layers.{d}.attn_norms.{d}.scale", .{ li, ai }), d);
                a.qkv = try w.get(tensorName(&buf, "transformer.layers.{d}.attns.{d}.wQKV.weight", .{ li, ai }), 3 * d * d);
                a.out = try w.get(tensorName(&buf, "transformer.layers.{d}.attns.{d}.wO.weight", .{ li, ai }), d * d);
                a.gate = if (std.mem.eql(u8, gate_kind, "sdpa_sigmoid"))
                    try w.get(tensorName(&buf, "transformer.layers.{d}.attns.{d}._gate.weight", .{ li, ai }), d * d)
                else
                    null;
                layer.attns[ai] = a;
            }
            layer.mlp_norm = try w.get(tensorName(&buf, "transformer.layers.{d}.mlp_norm.scale", .{li}), d);
            layer.w12 = try w.get(tensorName(&buf, "transformer.layers.{d}.mlp.w12.weight", .{li}), 2 * hid * d);
            layer.w3 = try w.get(tensorName(&buf, "transformer.layers.{d}.mlp.w3.weight", .{li}), d * hid);
            layers[li] = layer;
        }

        const hd: usize = config.headDim();
        const rd: usize = config.ropeDim();
        const inv_freq = try allocator.alloc(f32, hd / 2);
        @memset(inv_freq, 0);
        const base: f32 = @floatCast(config.rope_theta);
        for (0..rd / 2) |i| {
            const idx: f32 = @floatFromInt(2 * i);
            inv_freq[i] = 1.0 / std.math.pow(f32, base, idx / @as(f32, @floatFromInt(rd)));
        }

        return .{
            .config = config,
            .patch_proj = patch_proj,
            .forecast_query = forecast_query,
            .layers = layers,
            .out_norm = out_norm,
            .head = head,
            .inv_freq = inv_freq,
        };
    }

    pub fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.layers);
        allocator.free(self.inv_freq);
    }

    /// Per-patch causal `(loc, scale)` of packed `(R, T)` rows, each `(R, T / patch_size)`.
    /// Each patch uses the running mean/std (float64 Welford) of everything observed up to
    /// its last step, then both are rounded to float32.
    pub fn patchStats(self: *const Model, loc: []f32, scale: []f32, data: []const f32, observed: []const bool, rows: usize, t: usize) void {
        const job: StatsJob = .{ .model = self, .loc = loc, .scale = scale, .data = data, .observed = observed, .t = t };
        par.forRangeMin(rows, 4, &job, StatsJob.run);
    }

    /// Quantiles `(T, h * patch_size, Q)` in data units for the target rows after the packed
    /// context, `T` being the number of target rows (series by series, in `roles` order).
    pub fn predictBlock(self: *const Model, allocator: Allocator, out: []f32, in: BlockInput) !void {
        const c = self.config;
        const ps: usize = c.patch_size;
        const d: usize = c.d_model;
        const q: usize = c.numQuantiles();
        const t = in.t;
        const rows = in.ctx.len / t;
        const vall = in.roles.len;
        const h = in.h;
        const l = t / ps;
        const hl = h * ps;

        var cov = false;
        var n_tgt: usize = 0;
        for (in.roles) |role| {
            cov = cov or role != .target;
            if (role == .target) n_tgt += 1;
        }
        if (!cov and in.known_future != null) return error.InvalidInput;
        if (cov and (h < 1 or h > c.block_size)) return error.BlockTooLong;

        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // The window: the context, plus (with covariates) the horizon blanks / known futures.
        const w = if (cov) t + hl else t;
        const n = w / ps;
        var full = in.ctx;
        var full_obs = in.obs;
        if (cov) {
            const f = try arena.alloc(f32, rows * w);
            const fo = try arena.alloc(bool, rows * w);
            for (0..rows) |r| {
                @memcpy(f[r * w ..][0..t], in.ctx[r * t ..][0..t]);
                @memcpy(fo[r * w ..][0..t], in.obs[r * t ..][0..t]);
                if (in.known_future) |kf| {
                    @memcpy(f[r * w + t ..][0..hl], kf[r * hl ..][0..hl]);
                    @memcpy(fo[r * w + t ..][0..hl], in.known_future_obs.?[r * hl ..][0..hl]);
                } else {
                    @memset(f[r * w + t ..][0..hl], 0);
                    @memset(fo[r * w + t ..][0..hl], false);
                }
            }
            full = f;
            full_obs = fo;
        }

        // Causal per-patch normalization.
        const loc = try arena.alloc(f32, rows * n);
        const scale = try arena.alloc(f32, rows * n);
        self.patchStats(loc, scale, full, full_obs, rows, w);
        const eps32 = std.math.floatEps(f32);
        const emb_in = try arena.alloc(f32, rows * n * 2 * ps);
        for (0..rows) |r| {
            const known = cov and in.roles[r % vall] == .known;
            // Known-future covariates use whole-row statistics; 0/1 indicator rows skip the asinh.
            var kloc: f32 = 0;
            var kscale: f32 = 1;
            var binary = false;
            if (known) {
                var cnt: f32 = 0;
                var sum: f32 = 0;
                var zero_one = true;
                var any = false;
                for (0..w) |i| if (full_obs[r * w + i]) {
                    const x = full[r * w + i];
                    cnt += 1;
                    sum += x;
                    zero_one = zero_one and (x == 0 or x == 1);
                    any = true;
                };
                kloc = sum / @max(cnt, 1.0);
                var ss: f32 = 0;
                for (0..w) |i| if (full_obs[r * w + i]) {
                    const dv = full[r * w + i] - kloc;
                    ss += dv * dv;
                };
                kscale = @max(@sqrt(ss / @max(cnt, 1.0)), @as(f32, @floatCast(c.minimum_scale)));
                binary = zero_one and any;
                if (binary) {
                    kloc = 0;
                    kscale = 1;
                }
            }
            for (0..n) |p| {
                const lo = if (known) kloc else loc[r * n + p];
                const sc = @max(if (known) kscale else scale[r * n + p], eps32);
                const dst = emb_in[(r * n + p) * 2 * ps ..][0 .. 2 * ps];
                for (0..ps) |i| {
                    const obs = full_obs[r * w + p * ps + i];
                    var v: f32 = 0;
                    if (obs) {
                        v = (full[r * w + p * ps + i] - lo) / sc;
                        if (c.use_asinh and !(known and binary)) v = std.math.asinh(v);
                    }
                    dst[i] = v;
                    dst[ps + i] = if (obs) 0 else 1;
                }
            }
        }

        // Tokens: [Enc ; Dec]. Encoder tokens cover the whole window with covariates, else the
        // context. Only the decoder tokens of the horizon are materialized: the others read only
        // themselves and nothing reads them.
        const ne = n;
        const sl = ne + h;
        const x = try arena.alloc(f32, rows * sl * d);
        const enc = try arena.alloc(f32, rows * ne * d);
        try self.patch_proj.forward(arena, enc, emb_in, rows * ne);
        for (0..rows) |r| {
            @memcpy(x[r * sl * d ..][0 .. ne * d], enc[r * ne * d ..][0 .. ne * d]);
            for (0..h) |j| @memcpy(x[(r * sl + ne + j) * d ..][0..d], self.forecast_query);
        }

        // Key validity per (row, patch); first observed patch per series.
        const key_valid = try arena.alloc(bool, rows * ne);
        const first_obs = try arena.alloc(usize, in.groups);
        @memset(first_obs, ne);
        for (0..rows) |r| {
            const g = r / vall;
            var started = false;
            for (0..ne) |p| {
                var obs = false;
                for (0..ps) |i| obs = obs or full_obs[r * w + p * ps + i];
                if (obs) first_obs[g] = @min(first_obs[g], p);
                started = started or obs;
                key_valid[r * ne + p] = if (c.attend_missing_patches) (obs or started) else obs;
            }
        }
        for (first_obs) |*f| if (f.* == ne) {
            f.* = 0;
        };

        const geo: Geo = .{
            .rows = rows,
            .groups = in.groups,
            .vall = vall,
            .ne = ne,
            .l = l,
            .h = h,
            .roles = in.roles,
            .cov = cov,
            .key_valid = key_valid,
            .first_obs = first_obs,
        };
        try self.runTransformer(arena, x, geo);

        // Head on the target rows' decoder tokens.
        const n_rows_out = in.groups * n_tgt;
        const dec = try arena.alloc(f32, n_rows_out * h * d);
        var oi: usize = 0;
        for (0..rows) |r| {
            if (in.roles[r % vall] != .target) continue;
            nn.rmsNorm(dec[oi * h * d ..][0 .. h * d], x[(r * sl + ne) * d ..][0 .. h * d], h, d, self.out_norm, @floatCast(c.norm_eps));
            oi += 1;
        }
        const raw = try arena.alloc(f32, n_rows_out * hl * q);
        try self.head.forward(arena, raw, dec, n_rows_out * h);

        const m = c.medianIndex();
        const widths = try arena.alloc(f32, q);
        @memset(widths, 0);
        for (m + 1..q) |i| widths[i] = @floatCast(c.quantile_knots[i] - c.quantile_knots[i - 1]);
        for (0..m) |i| widths[i] = @floatCast(c.quantile_knots[i + 1] - c.quantile_knots[i]);
        const wf: f32 = @floatCast(c.quantile_width_factor);
        const big = std.math.floatMax(f32);

        oi = 0;
        for (0..rows) |r| {
            if (in.roles[r % vall] != .target) continue;
            // Anchor frame: the last context patch.
            const lo = loc[r * n + l - 1];
            const sc = scale[r * n + l - 1];
            for (0..hl) |sidx| {
                const rv = raw[(oi * hl + sidx) * q ..][0..q];
                const ov = out[(oi * hl + sidx) * q ..][0..q];
                // Monotone head: median free, the rest fan out by cumulative softplus increments.
                const med = rv[m];
                var acc: f32 = 0;
                for (m + 1..q) |i| {
                    acc += wf * widths[i] * nn.softplus(rv[i]);
                    ov[i] = std.math.clamp(med + acc, -big, big);
                }
                acc = 0;
                var i: usize = m;
                while (i > 0) {
                    i -= 1;
                    acc += wf * widths[i] * nn.softplus(rv[i]);
                    ov[i] = std.math.clamp(med - acc, -big, big);
                }
                ov[m] = med;
                // Back to data units: sinh(x) * scale + loc, then fix any overflow.
                for (ov) |*v| {
                    const y = if (c.use_asinh) std.math.sinh(v.*) else v.*;
                    v.* = y * sc + lo;
                }
                nn.clampNonfinite(ov);
            }
            oi += 1;
        }
    }

    fn runTransformer(self: *const Model, arena: Allocator, x: []f32, geo: Geo) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const sl = geo.s();
        const rows = geo.rows;
        const eps: f32 = @floatCast(c.norm_eps);
        const hid: usize = c.mlp_hidden_dim;

        const xn = try arena.alloc(f32, rows * sl * d);
        const delta = try arena.alloc(f32, rows * sl * d);
        const mlp_h = try arena.alloc(f32, rows * sl * 2 * hid);
        const mlp_a = try arena.alloc(f32, rows * sl * hid);

        // RoPE tables per series: encoder token p sits at time p, decoder token j at time l + j,
        // both relative to the series' first observed patch.
        const hd: usize = c.headDim();
        const half = hd / 2;
        const cos_t = try arena.alloc(f32, geo.groups * sl * half);
        const sin_t = try arena.alloc(f32, geo.groups * sl * half);
        for (0..geo.groups) |g| {
            for (0..sl) |tok| {
                const time: i64 = if (tok < geo.ne) @intCast(tok) else @intCast(geo.l + tok - geo.ne);
                const pos: f32 = @floatFromInt(time - @as(i64, @intCast(geo.first_obs[g])));
                for (0..half) |i| {
                    const ang = self.inv_freq[i] * pos;
                    cos_t[(g * sl + tok) * half + i] = @cos(ang);
                    sin_t[(g * sl + tok) * half + i] = @sin(ang);
                }
            }
        }

        for (self.layers) |layer| {
            for (layer.attns[0..layer.n_attn]) |a| {
                nn.rmsNorm(xn, x, rows * sl, d, a.norm, eps);
                if (a.is_time) {
                    try self.timeAttention(arena, delta, xn, a, geo, cos_t, sin_t);
                } else {
                    try self.variateAttention(arena, delta, xn, a, geo);
                }
                for (x, delta[0..x.len]) |*xv, dv| xv.* += dv;
            }
            nn.rmsNorm(xn, x, rows * sl, d, layer.mlp_norm, eps);
            nn.linear(mlp_h, xn, rows * sl, layer.w12, 2 * hid, d, null);
            const act: MlpAct = .{ .h = mlp_h, .a = mlp_a, .hid = hid };
            par.forRangeMin(rows * sl, 32, &act, MlpAct.run);
            nn.linear(delta, mlp_a, rows * sl, layer.w3, d, hid, null);
            for (x, delta[0..x.len]) |*xv, dv| xv.* += dv;
        }
    }

    /// Time attention per row over `[Enc ; Dec]`. Encoder queries read valid encoder keys
    /// causally (diagonal always open); known-future rows read all valid keys. Decoder queries
    /// read the valid context keys and their own causal decoder chain.
    fn timeAttention(self: *const Model, arena: Allocator, out: []f32, xn: []const f32, a: Attn, geo: Geo, cos_t: []const f32, sin_t: []const f32) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const rows = geo.rows;
        const sl = geo.s();

        const qkv = try arena.alloc(f32, rows * sl * 3 * d);
        nn.linear(qkv, xn, rows * sl, a.qkv, 3 * d, d, null);
        const qa = try arena.alloc(f32, rows * sl * d);
        const ka = try arena.alloc(f32, rows * sl * d);
        const va = try arena.alloc(f32, rows * sl * d);
        const oa = try arena.alloc(f32, rows * sl * d);

        const gate = if (a.gate) |gw| blk: {
            const g = try arena.alloc(f32, rows * sl * d);
            nn.linear(g, xn, rows * sl, gw, d, d, null);
            break :blk g;
        } else null;

        const job: TimeJob = .{
            .model = self,
            .geo = geo,
            .qkv = qkv,
            .q = qa,
            .k = ka,
            .v = va,
            .o = oa,
            .cos_t = cos_t,
            .sin_t = sin_t,
        };
        // Phase 1: split the fused projection into heads and apply RoPE, per token.
        par.forRangeMin(rows * sl, 16, &job, TimeJob.prepare);
        // Phase 2: attention per (row, head).
        par.forRange(rows * c.num_heads, &job, TimeJob.attend);
        // Phase 3: gate, then the output projection.
        if (gate) |gt| {
            const gj: GateJob = .{ .dst = out, .x = oa, .g = gt, .per = sl * d };
            par.forRange(rows, &gj, GateJob.run);
        } else @memcpy(out[0 .. rows * sl * d], oa);
        const proj = try arena.alloc(f32, rows * sl * d);
        nn.linear(proj, out[0 .. rows * sl * d], rows * sl, a.out, d, d, null);
        @memcpy(out[0 .. rows * sl * d], proj);
    }

    /// Variate attention: at every token slot, the rows of a series attend each other. A row
    /// reads another row on an encoder slot only if that row's patch is a valid key. With
    /// covariates two firewalls apply: a covariate row never reads a target row, and a
    /// known-future row never reads a causal row. The diagonal is always open.
    fn variateAttention(self: *const Model, arena: Allocator, out: []f32, xn: []const f32, a: Attn, geo: Geo) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const rows = geo.rows;
        const sl = geo.s();

        const qkv = try arena.alloc(f32, rows * sl * 3 * d);
        nn.linear(qkv, xn, rows * sl, a.qkv, 3 * d, d, null);
        const gate = if (a.gate) |gw| blk: {
            const g = try arena.alloc(f32, rows * sl * d);
            nn.linear(g, xn, rows * sl, gw, d, d, null);
            break :blk g;
        } else null;

        const o = try arena.alloc(f32, rows * sl * d);
        const job: VariateJob = .{ .model = self, .geo = geo, .qkv = qkv, .o = o };
        par.forRangeMin(geo.groups * sl, 8, &job, VariateJob.run);
        if (gate) |gt| nn.mulSigmoid(o, o, gt);
        nn.linear(out[0 .. rows * sl * d], o, rows * sl, a.out, d, d, null);
    }
};

const GateJob = struct {
    dst: []f32,
    x: []const f32,
    g: []const f32,
    per: usize,

    fn run(self: *const GateJob, r0: usize, r1: usize) void {
        for (r0..r1) |r| nn.mulSigmoid(self.dst[r * self.per ..][0..self.per], self.x[r * self.per ..][0..self.per], self.g[r * self.per ..][0..self.per]);
    }
};

const TimeJob = struct {
    model: *const Model,
    geo: Geo,
    qkv: []const f32,
    q: []f32,
    k: []f32,
    v: []f32,
    o: []f32,
    cos_t: []const f32,
    sin_t: []const f32,

    /// Tokens `[t0, t1)` of the packed `(row, token)` sequence: the fused weight is laid out
    /// `(qkv, head_dim, heads)`; gather into `(token, head, head_dim)` and rotate q and k.
    /// Partial RoPE: dims past rope_dim have zero frequency, i.e. the identity.
    fn prepare(self: *const TimeJob, t0: usize, t1: usize) void {
        const c = self.model.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const half = hd / 2;
        const sl = self.geo.s();
        for (t0..t1) |idx| {
            const r = idx / sl;
            const tok = idx % sl;
            const g = r / self.geo.vall;
            const src = self.qkv[idx * 3 * d ..][0 .. 3 * d];
            for (0..nh) |hh| {
                for (0..hd) |dd| {
                    self.q[idx * d + hh * hd + dd] = src[0 * d + dd * nh + hh];
                    self.k[idx * d + hh * hd + dd] = src[1 * d + dd * nh + hh];
                    self.v[idx * d + hh * hd + dd] = src[2 * d + dd * nh + hh];
                }
                const qh = self.q[idx * d + hh * hd ..][0..hd];
                const kh = self.k[idx * d + hh * hd ..][0..hd];
                for (0..half) |i| {
                    const cs = self.cos_t[(g * sl + tok) * half + i];
                    const sn = self.sin_t[(g * sl + tok) * half + i];
                    const q1 = qh[i];
                    const q2 = qh[i + half];
                    qh[i] = q1 * cs - q2 * sn;
                    qh[i + half] = q2 * cs + q1 * sn;
                    const k1 = kh[i];
                    const k2 = kh[i + half];
                    kh[i] = k1 * cs - k2 * sn;
                    kh[i + half] = k2 * cs + k1 * sn;
                }
            }
        }
    }

    /// Items `[first, last)` of `(row, head)`.
    fn attend(self: *const TimeJob, first: usize, last: usize) void {
        const c = self.model.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const geo = self.geo;
        const ne = geo.ne;
        const l = geo.l;
        const h = geo.h;
        const sl = geo.s();
        const sm_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const ninf = -std.math.inf(f32);

        const scores = std.heap.smp_allocator.alloc(f32, @max(ne * ne, h * sl)) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(scores);

        for (first..last) |item| {
            const r = item / nh;
            const hh = item % nh;
            const off = hh * hd;
            const bidir = geo.cov and geo.role(r) == .known;
            const base = r * sl * d;
            const q = self.q[base..][0 .. sl * d];
            const k = self.k[base..][0 .. sl * d];
            const v = self.v[base..][0 .. sl * d];
            const o = self.o[base..][0 .. sl * d];
            const kv = geo.key_valid[r * ne ..][0..ne];

            // Encoder queries over encoder keys.
            nn.gemmNT(scores, ne, q[off..], d, k[off..], d, ne, ne, hd, sm_scale);
            for (0..ne) |i| {
                const row = scores[i * ne ..][0..ne];
                for (0..ne) |j| {
                    const allowed = j == i or (kv[j] and (bidir or j <= i));
                    if (!allowed) row[j] = ninf;
                }
                nn.softmax(row);
            }
            nn.gemmNN(o[off..], d, scores, ne, v[off..], d, ne, hd, ne);
            // Decoder queries: valid context keys, then the causal decoder chain.
            nn.gemmNT(scores, sl, q[ne * d + off ..], d, k[off..], d, h, sl, hd, sm_scale);
            for (0..h) |j| {
                const row = scores[j * sl ..][0..sl];
                for (0..ne) |kk| if (kk >= l or !kv[kk]) {
                    row[kk] = ninf;
                };
                for (ne + j + 1..sl) |kk| row[kk] = ninf;
                nn.softmax(row);
            }
            nn.gemmNN(o[ne * d + off ..], d, scores, sl, v[off..], d, h, hd, sl);
        }
    }
};

const VariateJob = struct {
    model: *const Model,
    geo: Geo,
    qkv: []const f32,
    o: []f32,

    /// Items `[first, last)` of `(series, token slot)`.
    fn run(self: *const VariateJob, first: usize, last: usize) void {
        const c = self.model.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const geo = self.geo;
        const vall = geo.vall;
        const ne = geo.ne;
        const sl = geo.s();
        const sm_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const sc = std.heap.smp_allocator.alloc(f32, vall) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(sc);

        for (first..last) |item| {
            const g = item / sl;
            const tok = item % sl;
            const r0 = g * vall;
            for (0..nh) |hh| {
                for (0..vall) |iq| {
                    const rq = r0 + iq;
                    const qsrc = self.qkv[(rq * sl + tok) * 3 * d ..][0 .. 3 * d];
                    for (0..vall) |ik| {
                        const rk = r0 + ik;
                        var allowed = ik == iq;
                        if (!allowed) {
                            const valid = tok >= ne or geo.key_valid[rk * ne + tok];
                            var forbidden = false;
                            if (geo.cov) {
                                const rq_role = geo.roles[iq];
                                const rk_role = geo.roles[ik];
                                forbidden = (rq_role != .target and rk_role == .target) or (rq_role == .known and rk_role != .known);
                            }
                            allowed = valid and !forbidden;
                        }
                        if (!allowed) {
                            sc[ik] = -std.math.inf(f32);
                            continue;
                        }
                        const ksrc = self.qkv[(rk * sl + tok) * 3 * d ..][0 .. 3 * d];
                        var dot: f32 = 0;
                        for (0..hd) |dd| dot += qsrc[dd * nh + hh] * ksrc[d + dd * nh + hh];
                        sc[ik] = dot * sm_scale;
                    }
                    nn.softmax(sc);
                    const dst = self.o[(rq * sl + tok) * d + hh * hd ..][0..hd];
                    @memset(dst, 0);
                    for (0..vall) |ik| {
                        if (sc[ik] == 0) continue;
                        const vsrc = self.qkv[((r0 + ik) * sl + tok) * 3 * d + 2 * d ..][0..d];
                        for (0..hd) |dd| dst[dd] += sc[ik] * vsrc[dd * nh + hh];
                    }
                }
            }
        }
    }
};

const StatsJob = struct {
    model: *const Model,
    loc: []f32,
    scale: []f32,
    data: []const f32,
    observed: []const bool,
    t: usize,

    /// Rows `[r0, r1)`.
    fn run(self: *const StatsJob, r0: usize, r1: usize) void {
        const ps: usize = self.model.config.patch_size;
        const t = self.t;
        const n = t / ps;
        const min_scale = self.model.config.minimum_scale;
        for (r0..r1) |r| {
            var cum_w: f64 = 0;
            var cum_x: f64 = 0;
            var m2: f64 = 0;
            var prev_mean: f64 = 0;
            for (0..t) |i| {
                const w: f64 = if (self.observed[r * t + i]) 1.0 else 0.0;
                const x: f64 = self.data[r * t + i];
                cum_w += w;
                cum_x += w * x;
                const den = @max(cum_w, 1.0);
                const mean = cum_x / den;
                m2 += (x - prev_mean) * (x - mean) * w;
                prev_mean = mean;
                if (i % ps == ps - 1) {
                    const variance = m2 / @max(den - 1.0, 1.0);
                    self.loc[r * n + i / ps] = @floatCast(mean);
                    self.scale[r * n + i / ps] = @floatCast(@max(@sqrt(variance), min_scale));
                }
            }
        }
    }
};
