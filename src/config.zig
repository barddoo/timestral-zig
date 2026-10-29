//! Inference configuration, mirroring `timestral.config.TimestralConfig`.

const std = @import("std");

pub const Config = struct {
    patch_size: u32,
    d_model: u32,
    num_heads: u32,
    mlp_hidden_dim: u32,
    block_schedule: []const []const u8,
    quantile_knots: []const f64,
    block_size: u32,
    context_length: u32,
    rope_theta: f64 = 10000.0,
    rope_partial_factor: f64 = 0.5,
    norm_eps: f64 = 1e-6,
    attn_bias: bool = false,
    mlp_bias: bool = false,
    time_attn_gate: []const u8 = "sdpa_sigmoid",
    variate_attn_gate: []const u8 = "sdpa_sigmoid",
    minimum_scale: f64 = 1e-6,
    use_asinh: bool = true,
    stabilize_with_global: bool = false,
    covariate_enabled: bool = true,
    split_context_horizon_attn: bool = true,
    attend_missing_patches: bool = true,
    quantile_width_factor: f64 = 2.0,

    pub fn headDim(self: Config) u32 {
        return self.d_model / self.num_heads;
    }

    pub fn ropeDim(self: Config) u32 {
        return @intFromFloat(@as(f64, @floatFromInt(self.headDim())) * self.rope_partial_factor);
    }

    pub fn numQuantiles(self: Config) usize {
        return self.quantile_knots.len;
    }

    pub fn medianIndex(self: Config) usize {
        for (self.quantile_knots, 0..) |q, i| if (q == 0.5) return i;
        unreachable; // validate() guarantees a 0.5 knot
    }

    /// Whether the layer schedule can route a known covariate's future to the targets: some time
    /// op must be followed by at least one more time op and at least one variate op.
    pub fn supportsKnownCovariates(self: Config) bool {
        if (!self.covariate_enabled) return false;
        var ops: [256]bool = undefined;
        var n: usize = 0;
        for (self.block_schedule) |kind| {
            const ax = layerAxes(kind) catch continue;
            for (ax.is_time[0..ax.len]) |t| {
                if (n < ops.len) ops[n] = t;
                n += 1;
            }
        }
        n = @min(n, ops.len);
        for (0..n) |i| {
            if (!ops[i]) continue;
            var any_time = false;
            var any_variate = false;
            for (ops[i + 1 .. n]) |t| {
                any_time = any_time or t;
                any_variate = any_variate or !t;
            }
            if (any_time and any_variate) return true;
        }
        return false;
    }

    pub fn validate(self: Config) !void {
        if (self.d_model % self.num_heads != 0) return error.InvalidConfig;
        const rd = self.ropeDim();
        if (rd == 0 or rd % 2 != 0) return error.InvalidConfig;
        if (self.stabilize_with_global) return error.InvalidConfig;
        if (self.attn_bias or self.mlp_bias) return error.UnsupportedBias;
        for (self.block_schedule) |kind| _ = try layerAxes(kind);
        for (self.quantile_knots[1..], 0..) |q, i| if (!(self.quantile_knots[i] < q)) return error.InvalidConfig;
        var has_median = false;
        for (self.quantile_knots) |q| has_median = has_median or q == 0.5;
        if (!has_median) return error.InvalidConfig;
    }
};

pub const Axes = struct {
    /// True = time attention, false = variate attention, in the order the layer runs them.
    is_time: [2]bool,
    len: u2,
};

pub fn layerAxes(kind: []const u8) !Axes {
    if (std.mem.eql(u8, kind, "time")) return .{ .is_time = .{ true, false }, .len = 1 };
    if (std.mem.eql(u8, kind, "variate")) return .{ .is_time = .{ false, false }, .len = 1 };
    if (std.mem.eql(u8, kind, "time_variate")) return .{ .is_time = .{ true, false }, .len = 2 };
    if (std.mem.eql(u8, kind, "variate_time")) return .{ .is_time = .{ false, true }, .len = 2 };
    return error.UnknownLayerKind;
}

pub const Parsed = std.json.Parsed(Config);

pub fn parse(allocator: std.mem.Allocator, text: []const u8) !Parsed {
    const parsed = try std.json.parseFromSlice(Config, allocator, text, .{ .ignore_unknown_fields = true });
    errdefer parsed.deinit();
    try parsed.value.validate();
    return parsed;
}
