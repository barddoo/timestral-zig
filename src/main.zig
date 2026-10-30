//! timestral: zero-shot probabilistic time series forecasting from a Timestral checkpoint.

const std = @import("std");
const sys = @import("sys.zig");
const cfg = @import("config.zig");
const st = @import("safetensors.zig");
const model_mod = @import("model.zig");
const forecaster = @import("forecaster.zig");
const hub = @import("hub.zig");
const par = @import("par.zig");

const usage =
    \\usage: timestral <model> <series.csv> --horizon N [options]
    \\
    \\  <model>       a directory with config.json and model.safetensors, or a Hugging Face
    \\                Hub repo id such as mistralai/Timestral-28M (cached, else downloaded)
    \\  <series.csv>  one line per variate; values separated by commas or whitespace; nan / NA /
    \\                an empty field marks a missing value. Separate several series with a
    \\                blank line (all with the same number of variates and steps).
    \\
    \\covariates (same blank-line-separated blocks, one block per series, one line per covariate):
    \\  --past FILE         past-only covariates, as many steps as the series
    \\  --known FILE        known-future covariates, steps + horizon values per line
    \\  --cat-past FILE     categorical codes, as many steps as the series
    \\  --cat-known FILE    categorical codes, steps + horizon values per line
    \\
    \\options:
    \\  --horizon N         steps to forecast (required)
    \\  --max-context N     keep only the most recent N steps (default: trained length)
    \\  --block-patches N   patches decoded per block (default 32)
    \\  --no-symmetric      skip averaging with the reflected forecast of -context
    \\  --no-clip           do not clamp non-negative series at 0
    \\  --revision REV      Hub branch, tag or commit (default main)
    \\  --offline           never touch the network (also HF_HUB_OFFLINE=1)
    \\  --threads N         worker threads (default: one per logical CPU; 1 = single-threaded)
    \\  --time              print model load and forecast wall time to stderr
    \\
    \\prints CSV: series,variate,step,q<level>... (step 1 is the first forecast step)
    \\
;

const Args = struct {
    model: []const u8,
    series: []const u8,
    past: ?[]const u8 = null,
    known: ?[]const u8 = null,
    cat_past: ?[]const u8 = null,
    cat_known: ?[]const u8 = null,
    horizon: usize = 0,
    revision: []const u8 = "main",
    offline: bool = false,
    time: bool = false,
    threads: usize = 0, // 0 = one per logical CPU
    opts: forecaster.Options = .{},
};

fn parseArgs(argv: []const [:0]const u8, env: *const std.process.Environ.Map) !Args {
    var positional: [2][]const u8 = undefined;
    var npos: usize = 0;
    var args: Args = .{ .model = "", .series = "" };
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a: []const u8 = argv[i];
        const takes_number = std.mem.eql(u8, a, "--horizon") or std.mem.eql(u8, a, "--threads") or std.mem.eql(u8, a, "--max-context") or std.mem.eql(u8, a, "--block-patches");
        const takes_string = std.mem.eql(u8, a, "--past") or std.mem.eql(u8, a, "--known") or std.mem.eql(u8, a, "--cat-past") or
            std.mem.eql(u8, a, "--cat-known") or std.mem.eql(u8, a, "--revision");
        if (takes_number or takes_string) {
            i += 1;
            if (i >= argv.len) return error.MissingValue;
            const sval: []const u8 = argv[i];
            if (takes_number) {
                const val = std.fmt.parseInt(usize, sval, 10) catch return error.InvalidNumber;
                if (std.mem.eql(u8, a, "--horizon")) args.horizon = val;
                if (std.mem.eql(u8, a, "--threads")) args.threads = val;
                if (std.mem.eql(u8, a, "--max-context")) args.opts.max_context = val;
                if (std.mem.eql(u8, a, "--block-patches")) args.opts.per_shot_horizon_patches = @max(val, 1);
            } else if (std.mem.eql(u8, a, "--past")) {
                args.past = sval;
            } else if (std.mem.eql(u8, a, "--known")) {
                args.known = sval;
            } else if (std.mem.eql(u8, a, "--cat-past")) {
                args.cat_past = sval;
            } else if (std.mem.eql(u8, a, "--cat-known")) {
                args.cat_known = sval;
            } else args.revision = sval;
        } else if (std.mem.eql(u8, a, "--no-symmetric")) {
            args.opts.symmetric_averaging = false;
        } else if (std.mem.eql(u8, a, "--no-clip")) {
            args.opts.clip_nonnegative = false;
        } else if (std.mem.eql(u8, a, "--offline")) {
            args.offline = true;
        } else if (std.mem.eql(u8, a, "--time")) {
            args.time = true;
        } else if (a.len > 0 and a[0] == '-') {
            return error.UnknownOption;
        } else {
            if (npos == 2) return error.TooManyArguments;
            positional[npos] = argv[i];
            npos += 1;
        }
    }
    if (npos != 2 or args.horizon == 0) return error.MissingArguments;
    args.model = positional[0];
    args.series = positional[1];
    if (env.get("HF_HUB_OFFLINE")) |v| {
        if (std.mem.eql(u8, v, "1")) args.offline = true;
    }
    return args;
}

/// A file of blank-line-separated blocks of equal-length rows, as `(blocks, rows, len)` float64.
const Blocks = struct {
    data: []f64,
    blocks: usize,
    rows: usize,
    len: usize,
};

fn parseValue(tok: []const u8) f64 {
    const s = std.mem.trim(u8, tok, " \t\r");
    if (s.len == 0) return std.math.nan(f64);
    return std.fmt.parseFloat(f64, s) catch std.math.nan(f64);
}

fn parseBlocks(allocator: std.mem.Allocator, text: []const u8) !Blocks {
    var vals: std.ArrayList(f64) = .empty;
    errdefer vals.deinit(allocator);
    var blocks: usize = 0;
    var rows_in_block: usize = 0;
    var rows: usize = 0;
    var len: usize = 0;
    var have_len = false;

    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len > 0 and line[0] == '#') continue;
        if (line.len == 0) {
            // A blank line closes the current block.
            if (rows_in_block > 0) {
                if (blocks == 0) rows = rows_in_block else if (rows_in_block != rows) return error.RaggedBlocks;
                blocks += 1;
                rows_in_block = 0;
            }
            continue;
        }
        const before = vals.items.len;
        if (std.mem.indexOfScalar(u8, line, ',') != null) {
            var f = std.mem.splitScalar(u8, line, ',');
            while (f.next()) |tok| try vals.append(allocator, parseValue(tok));
        } else {
            var f = std.mem.tokenizeAny(u8, line, " \t");
            while (f.next()) |tok| try vals.append(allocator, parseValue(tok));
        }
        const n = vals.items.len - before;
        if (!have_len) {
            len = n;
            have_len = true;
        } else if (n != len) {
            return error.RaggedInput;
        }
        rows_in_block += 1;
    }
    if (rows_in_block > 0) {
        if (blocks == 0) rows = rows_in_block else if (rows_in_block != rows) return error.RaggedBlocks;
        blocks += 1;
    }
    if (blocks == 0 or len == 0) return error.EmptyInput;
    return .{ .data = try vals.toOwnedSlice(allocator), .blocks = blocks, .rows = rows, .len = len };
}

fn loadBlocks(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Blocks {
    const text = try sys.readFile(io, allocator, path);
    defer allocator.free(text);
    return parseBlocks(allocator, text);
}

fn toF32(allocator: std.mem.Allocator, x: []const f64) ![]f32 {
    const out = try allocator.alloc(f32, x.len);
    for (out, x) |*o, v| o.* = @floatCast(v);
    return out;
}

/// A covariate file, checked against the series: one block per series, rows of `len` values.
fn covariateFile(io: std.Io, allocator: std.mem.Allocator, path: ?[]const u8, batch: usize, len: usize, what: []const u8) !?Blocks {
    const p = path orelse return null;
    const blk = try loadBlocks(io, allocator, p);
    if (blk.blocks != batch or blk.len != len) {
        sys.eprintf(io, "{s}: expected {d} series with {d} values per line, got {d} series with {d}\n", .{ what, batch, len, blk.blocks, blk.len });
        return error.CovariateShape;
    }
    return blk;
}

fn run(init: std.process.Init) !void {
    const allocator = init.gpa;

    var arg_iter = try init.minimal.args.iterateAllocator(allocator);
    defer arg_iter.deinit();
    var argv: std.ArrayList([:0]const u8) = .empty;
    defer argv.deinit(allocator);
    while (arg_iter.next()) |a| try argv.append(allocator, a);
    const args = parseArgs(argv.items, init.environ_map) catch |err| {
        sys.eprintf(init.io, "argument error: {s}\n", .{@errorName(err)});
        return err;
    };

    // Worker pool for `par.forRange` (this thread plus `threads - 1` workers); also the `Io`
    // behind file access and downloads.
    par.threads = if (args.threads == 0) (std.Thread.getCpuCount() catch 1) else args.threads;
    var threaded: std.Io.Threaded = .init(allocator, .{ .async_limit = .limited(par.threads - 1) });
    defer threaded.deinit();
    const io = threaded.io();
    const par_io: std.Io = if (par.threads > 1) io else std.Io.Threaded.global_single_threaded.io();
    const t_start = sys.now(io);

    const model_dir = hub.resolve(io, allocator, init.environ_map, args.model, args.revision, args.offline) catch |err| {
        sys.eprintf(io, "cannot find model '{s}' ({s})\n", .{ args.model, @errorName(err) });
        return err;
    };
    defer allocator.free(model_dir);

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    const cfg_text = try sys.readFile(io, allocator, cfg_path);
    defer allocator.free(cfg_text);
    const parsed = try cfg.parse(allocator, cfg_text);
    defer parsed.deinit();

    const w_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{model_dir});
    defer allocator.free(w_path);
    var weights = try st.Weights.open(io, allocator, w_path);
    defer weights.close(io);
    var model = try model_mod.Model.load(io, allocator, parsed.value, &weights);
    defer model.deinit(allocator);
    const t_loaded = sys.now(io);

    const series = try loadBlocks(io, allocator, args.series);
    defer allocator.free(series.data);
    const b = series.blocks;
    const v = series.rows;
    const t = series.len;

    const values = try toF32(allocator, series.data);
    defer allocator.free(values);
    const observed = try allocator.alloc(bool, values.len);
    defer allocator.free(observed);
    for (values, observed) |*x, *o| {
        o.* = std.math.isFinite(x.*);
        if (!o.*) x.* = 0;
    }

    var cov: forecaster.Covariates = .{};
    const past = try covariateFile(io, allocator, args.past, b, t, "--past");
    const known = try covariateFile(io, allocator, args.known, b, t + args.horizon, "--known");
    const cat_past = try covariateFile(io, allocator, args.cat_past, b, t, "--cat-past");
    const cat_known = try covariateFile(io, allocator, args.cat_known, b, t + args.horizon, "--cat-known");
    defer for ([_]?Blocks{ past, known, cat_past, cat_known }) |blk| if (blk) |x| allocator.free(x.data);
    const past32 = if (past) |p| try toF32(allocator, p.data) else null;
    defer if (past32) |x| allocator.free(x);
    const known32 = if (known) |k| try toF32(allocator, k.data) else null;
    defer if (known32) |x| allocator.free(x);
    if (past) |p| {
        cov.past = past32;
        cov.n_past = p.rows;
    }
    if (known) |k| {
        cov.known = known32;
        cov.n_known = k.rows;
    }
    if (cat_past) |p| {
        cov.cat_past = p.data;
        cov.n_cat_past = p.rows;
    }
    if (cat_known) |k| {
        cov.cat_known = k.data;
        cov.n_cat_known = k.rows;
    }

    const result = try forecaster.forecast(allocator, par_io, &model, values, observed, b, v, t, args.horizon, cov, args.opts);
    defer allocator.free(result);
    if (args.time) sys.eprintf(io, "load={d:.3}s forecast={d:.3}s\n", .{ t_loaded - t_start, sys.now(io) - t_loaded });

    const q = parsed.value.numQuantiles();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "series,variate,step");
    for (parsed.value.quantile_knots) |level| try out.print(allocator, ",q{d}", .{level});
    try out.append(allocator, '\n');
    for (0..b) |s| {
        for (0..v) |r| {
            for (0..args.horizon) |step| {
                try out.print(allocator, "{d},{d},{d}", .{ s, r, step + 1 });
                for (result[(((s * v) + r) * args.horizon + step) * q ..][0..q]) |x| try out.print(allocator, ",{d}", .{x});
                try out.append(allocator, '\n');
            }
        }
    }
    try sys.writeStdout(io, out.items);
}

pub fn main(init: std.process.Init) u8 {
    run(init) catch |err| {
        switch (err) {
            error.MissingArguments, error.MissingValue, error.InvalidNumber, error.UnknownOption, error.TooManyArguments => sys.eprint(init.io, usage),
            else => sys.eprintf(init.io, "error: {s}\n", .{@errorName(err)}),
        }
        return 1;
    };
    return 0;
}
