//! Memory-mapped safetensors reader. F32 tensors are used in place; F16 and BF16 tensors are
//! converted to float32 on first access.

const std = @import("std");
const sys = @import("sys.zig");

pub const Weights = struct {
    allocator: std.mem.Allocator,
    converted: std.ArrayList([]f32) = .empty,
    map: sys.Mapped,
    header: std.json.Parsed(std.json.Value),
    data_start: usize,

    pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Weights {
        var map = try sys.Mapped.open(io, path);
        errdefer map.close(io);
        if (map.bytes.len < 8) return error.InvalidSafetensors;
        const hlen: usize = @intCast(std.mem.readInt(u64, map.bytes[0..8], .little));
        if (8 + hlen > map.bytes.len) return error.InvalidSafetensors;
        const header = try std.json.parseFromSlice(std.json.Value, allocator, map.bytes[8 .. 8 + hlen], .{});
        return .{ .allocator = allocator, .map = map, .header = header, .data_start = 8 + hlen };
    }

    pub fn close(self: *Weights, io: std.Io) void {
        for (self.converted.items) |buf| self.allocator.free(buf);
        self.converted.deinit(self.allocator);
        self.header.deinit();
        self.map.close(io);
    }

    /// The named float32 tensor, checked against the expected element count.
    pub fn get(self: *Weights, io: std.Io, name: []const u8, numel: usize) ![]const f32 {
        const entry = self.header.value.object.get(name) orelse {
            sys.eprintf(io, "missing tensor: {s}\n", .{name});
            return error.MissingTensor;
        };
        const obj = entry.object;
        const dtype = obj.get("dtype").?.string;
        const offs = obj.get("data_offsets").?.array.items;
        const begin: usize = @intCast(offs[0].integer);
        const end: usize = @intCast(offs[1].integer);
        const elem: usize = if (std.mem.eql(u8, dtype, "F32")) 4 else if (std.mem.eql(u8, dtype, "F16") or std.mem.eql(u8, dtype, "BF16")) 2 else {
            sys.eprintf(io, "tensor {s} has unsupported dtype {s}\n", .{ name, dtype });
            return error.UnsupportedDtype;
        };
        if (end - begin != numel * elem) {
            sys.eprintf(io, "tensor {s}: expected {d} elements, file has {d}\n", .{ name, numel, (end - begin) / elem });
            return error.ShapeMismatch;
        }
        const bytes = self.map.bytes[self.data_start + begin ..][0 .. end - begin];
        if (elem == 2) {
            const buf = try self.allocator.alloc(f32, numel);
            errdefer self.allocator.free(buf);
            const is_bf16 = dtype[0] == 'B';
            for (buf, 0..) |*o, i| {
                const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
                o.* = if (is_bf16) @bitCast(@as(u32, bits) << 16) else @floatCast(@as(f16, @bitCast(bits)));
            }
            try self.converted.append(self.allocator, buf);
            return buf;
        }
        const addr = @intFromPtr(bytes.ptr);
        if (addr % 4 != 0) return error.MisalignedTensor;
        const p: [*]const f32 = @ptrFromInt(addr);
        return p[0..numel];
    }
};
