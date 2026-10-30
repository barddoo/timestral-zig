//! Resolve a model argument to a directory with `config.json` and `model.safetensors`: a local
//! directory, a Hugging Face Hub repo id already in the Hub cache, or one downloaded with
//! `std.http.Client` into the same cache layout (`models--org--name/snapshots/<commit>/`) that
//! `huggingface_hub` and the `hf` CLI use.

const std = @import("std");
const sys = @import("sys.zig");

const Allocator = std.mem.Allocator;
const Dir = std.Io.Dir;
const files = [_][]const u8{ "config.json", "model.safetensors" };

fn cacheRoot(allocator: Allocator, env: *const std.process.Environ.Map) ![]u8 {
    if (env.get("HF_HUB_CACHE")) |p| if (p.len > 0) return allocator.dupe(u8, p);
    if (env.get("HF_HOME")) |p| if (p.len > 0) return std.fmt.allocPrint(allocator, "{s}/hub", .{p});
    const home = env.get("HOME") orelse return error.NoHomeDirectory;
    return std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/hub", .{home});
}

fn snapshotComplete(io: std.Io, allocator: Allocator, dir: []const u8) !bool {
    for (files) |f| {
        const p = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, f });
        defer allocator.free(p);
        if (!sys.exists(io, p)) return false;
    }
    return true;
}

/// A directory holding the model files. `offline` forbids network access.
pub fn resolve(io: std.Io, allocator: Allocator, env: *const std.process.Environ.Map, spec: []const u8, revision: []const u8, offline: bool) ![]u8 {
    if (try snapshotComplete(io, allocator, spec)) return allocator.dupe(u8, spec);

    // Anything else must be a Hub repo id, `org/name`.
    const slash = std.mem.indexOfScalar(u8, spec, '/') orelse return error.ModelNotFound;
    if (slash == 0 or slash + 1 == spec.len or std.mem.indexOfScalarPos(u8, spec, slash + 1, '/') != null) return error.ModelNotFound;

    const root = try cacheRoot(allocator, env);
    defer allocator.free(root);
    const repo_dir = try std.fmt.allocPrint(allocator, "{s}/models--{s}--{s}", .{ root, spec[0..slash], spec[slash + 1 ..] });
    defer allocator.free(repo_dir);

    // Cached: refs/<revision> names the commit (or the revision already is one).
    const ref_path = try std.fmt.allocPrint(allocator, "{s}/refs/{s}", .{ repo_dir, revision });
    defer allocator.free(ref_path);
    var commit: ?[]u8 = null;
    defer if (commit) |cm| allocator.free(cm);
    if (sys.readFile(io, allocator, ref_path)) |txt| {
        defer allocator.free(txt);
        const trimmed = std.mem.trim(u8, txt, " \r\n\t");
        if (trimmed.len > 0) commit = try allocator.dupe(u8, trimmed);
    } else |_| if (revision.len == 40) {
        commit = try allocator.dupe(u8, revision);
    }
    if (commit) |cm| {
        const snap = try std.fmt.allocPrint(allocator, "{s}/snapshots/{s}", .{ repo_dir, cm });
        if (try snapshotComplete(io, allocator, snap)) return snap;
        allocator.free(snap);
    }

    if (offline) return error.NotCachedOffline;
    return download(io, allocator, env, spec, revision, repo_dir);
}

/// GET `url` into `writer`, following redirects. The token is a privileged header, so it is only
/// sent to the original host and never to the CDN a download redirects to.
fn fetch(io: std.Io, allocator: Allocator, url: []const u8, writer: *std.Io.Writer, token: ?[]const u8) !void {
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    var auth_buf: [512]u8 = undefined;
    var headers: [1]std.http.Header = undefined;
    var privileged: []const std.http.Header = &.{};
    if (token) |tk| {
        headers[0] = .{ .name = "authorization", .value = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{tk}) };
        privileged = headers[0..1];
    }
    const res = try client.fetch(.{
        .location = .{ .url = url },
        .response_writer = writer,
        .privileged_headers = privileged,
        .extra_headers = &.{.{ .name = "user-agent", .value = "timestral-zig/0.1" }},
    });
    switch (res.status) {
        .ok => {},
        .unauthorized, .forbidden => {
            sys.eprint(io, "access denied: the repo is private or gated; set HF_TOKEN to a token with access\n");
            return error.AccessDenied;
        },
        .not_found => return error.RepoOrRevisionNotFound,
        else => {
            sys.eprintf(io, "unexpected HTTP status {d}\n", .{@backingInt(res.status)});
            return error.DownloadFailed;
        },
    }
}

fn download(io: std.Io, allocator: Allocator, env: *const std.process.Environ.Map, repo: []const u8, revision: []const u8, repo_dir: []const u8) ![]u8 {
    const token: ?[]const u8 = env.get("HF_TOKEN") orelse env.get("HUGGING_FACE_HUB_TOKEN");
    const cwd = Dir.cwd();
    try cwd.createDirPath(io, repo_dir);

    // The Hub API resolves the revision to a commit; every file is then fetched at that commit.
    var rev_enc: std.Io.Writer.Allocating = .init(allocator);
    defer rev_enc.deinit();
    for (revision) |ch| {
        if (ch == '/') try rev_enc.writer.writeAll("%2F") else try rev_enc.writer.writeByte(ch);
    }
    const api_url = try std.fmt.allocPrint(allocator, "https://huggingface.co/api/models/{s}/revision/{s}", .{ repo, rev_enc.written() });
    defer allocator.free(api_url);
    var api_body: std.Io.Writer.Allocating = .init(allocator);
    defer api_body.deinit();
    try fetch(io, allocator, api_url, &api_body.writer, token);
    const parsed = try std.json.parseFromSlice(struct { sha: []const u8 }, allocator, api_body.written(), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const commit = parsed.value.sha;

    const snap = try std.fmt.allocPrint(allocator, "{s}/snapshots/{s}", .{ repo_dir, commit });
    errdefer allocator.free(snap);
    try cwd.createDirPath(io, snap);

    for (files) |f| {
        const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/{s}/{s}", .{ repo, commit, f });
        defer allocator.free(url);
        const tmp = try std.fmt.allocPrint(allocator, "{s}/download-{s}.tmp", .{ repo_dir, f });
        defer allocator.free(tmp);
        const dest = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ snap, f });
        defer allocator.free(dest);

        sys.eprintf(io, "downloading {s}/{s} ...\n", .{ repo, f });
        {
            const out = try cwd.createFile(io, tmp, .{});
            defer out.close(io);
            errdefer cwd.deleteFile(io, tmp) catch {};
            var buf: [64 * 1024]u8 = undefined;
            var fw = out.writer(io, &buf);
            try fetch(io, allocator, url, &fw.interface, token);
            try fw.interface.flush();
        }
        try cwd.rename(tmp, cwd, dest, io);
    }

    // refs/<revision> -> commit, as huggingface_hub writes it.
    const refs_dir = try std.fmt.allocPrint(allocator, "{s}/refs", .{repo_dir});
    defer allocator.free(refs_dir);
    try cwd.createDirPath(io, refs_dir);
    const ref_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ refs_dir, revision });
    defer allocator.free(ref_path);
    try cwd.writeFile(io, .{ .sub_path = ref_path, .data = commit });
    return snap;
}
