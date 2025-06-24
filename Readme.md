# zflac

FLAC decoder implemented from [specifications](https://www.rfc-editor.org/rfc/rfc9639.html).

## Getting started

Add the library to your `build.zig.zon`:
```sh
zig fetch --save git+https://github.com/Senryoku/zflac
```
Declare the dependency in your `build.zig`:
```zig
const zflac = b.dependency("zflac", .{});
exe.root_module.addImport("zflac", zflac.module("zflac"));
```

## Usage

```zig
const zflac = @import("zflac");

pub fn main() !void {
	const allocator = std.heap.page_allocator;
	const file = try std.fs.cwd().openFile("music.flac", .{});
	defer file.close();

	// Prefer a buffered reader for performance.
    var buffered_reader = std.io.bufferedReader(file.reader());
    const reader = buffered_reader.reader();

	const decoded = try zflac.decode(allocator, reader); 
    defer decoded.deinit(allocator);

	// The returned structure holds some basic information on your file:
    std.debug.print("Channel count: {d}\n", .{decoded.channels});
    std.debug.print("Sample rate: {d}\n", .{decoded.sample_rate});
	// This is the number of significant bits and can be lower than the bit size of samples 
	// in the final array (e.g. 12bits samples are shifted left by 4 bits to 16bits).
    std.debug.print("Bits per samples: {d}\n", .{decoded.bits_per_sample}); 

	// `samples` is a tagged union based on the bit depth used in the file.
	switch (decoded.samples) {
		.s8 => |samples| play(i8, samples),
		.s16 => |samples| play(i16, samples),
		.s32 => |samples| play(i32, samples),
	}
}
```

See the `examples` folder for a more complete example.

## TODOs
 - Better handle signed/unsigned 8bit samples (Probably remove the i8 interface and auto convert to u8).
 - Add a mode to sync. to the next frame and start/resume decoding from there.