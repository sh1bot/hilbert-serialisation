# Pixel bus switching experiment

Count toggles on independent 8-bit sample lanes while transferring complete,
fixed-layout pixel chunks in raster, Morton/Z, or Hilbert order. No multiplexing,
bit significance weighting, lane permutation, reset load, or frame wraparound.

## Install

Python 3.10+:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Native JPEG decoding also requires the **libturbojpeg shared library**. On
Debian/Ubuntu install `libturbojpeg0` (or `libturbojpeg0t64` on newer releases);
on macOS `brew install jpeg-turbo`. If discovery fails, pass
`--turbojpeg-library /absolute/path/to/libturbojpeg.so` (or `.dylib`/DLL).
PyTurboJPEG and the unrelated Python package named turbojpeg must not be
installed in the same environment.

## Run

```sh
python pixel_bus.py photo.jpg --chunk 2x2 --chunk 4x4 --chunk 8x2 --lag 1 --lag 4 --json results.json
python pixel_bus.py picture.png --chunk 1x1 --chunk 4x2 --json results.json
python pixel_bus.py photo.jpg --mode rgb --chunk 1x1
```

`auto` (default) decodes JPEG into native Y/Cb/Cr planes using libjpeg-turbo,
without RGB conversion or chroma upsampling. Output calls Cb/Cr U/V. It preserves
4:4:4, 4:2:2, 4:2:0, 4:4:0, or 4:1:1 sampling; grayscale is Y only.
Unsupported native JPEG colorspaces fail explicitly; `--mode rgb` requests
Pillow RGB decoding instead. Native planes are decoded spatial samples after
inverse DCT, not compressed coefficients. Decoder rounding may vary by platform.

Other Pillow-supported inputs decode as RGB8. Alpha is ignored; only the first
frame is used. No YUV is synthesized from these inputs. Images are measured in
stored orientation, without applying EXIF rotation or ICC color management.
Non-8-bit inputs follow Pillow's RGB8 conversion behavior.

Chunk size is WIDTHxHEIGHT in full-resolution pixels. Default is 1x1 for RGB or
4:4:4, 2x1 for 4:2:2, and 2x2 for 4:2:0. Chunk dimensions must align with chroma
sampling. A 4:2:0 2x2 cycle has four Y, one U and one V sample: 48 independent
bits. RGB 2x2 has 96 bits. Each channel's sample positions use fixed row-major
lanes in every chunk. Partial edge chunks repeat the last sample; padded samples
are included in counts. JSON records dimensions, sampling, bus width and lane
shapes so these effects are visible.

All traversals visit the same rectangular chunk grid. Z/Hilbert use an enclosing
power-of-two square, skipping coordinates outside the grid. Hilbert's 2x2 base
order is top-left, bottom-left, bottom-right, top-right. Skipping cells in a
rectangle may break adjacency. Traversal keys are computed only for real chunks;
coordinates are sorted in memory, so very large images at 1x1 can be expensive.

For lag L, compare cycle t to cycle t-L using XOR, starting at t=L. Lag 1 measures
successive receiving-register updates. Lag L models overwrites of L rotating
storage slots during continuous writes; it does not model all stages of a shifting
FIFO. If L >= the number of cycles, there are zero measured comparisons.

The console shows total and per-channel toggles and percentage reduction versus
raster (negative means more toggles). JSON additionally gives each sample lane's
eight counts, LSB first, aggregates by channel bit position, and the fraction of
eligible bit updates that toggled. An undefined ratio is null, including when
raster has zero transitions. Every bit has equal weight. Counts cover data lanes,
not address/control logic, glitches, capacitance, or decoder switching.

## Tests

```sh
python -m unittest -v
# Also exercise native JPEG 4:4:4 / 4:2:2 / 4:2:0 decoding:
TEST_TURBOJPEG_LIBRARY=/path/to/libturbojpeg.so python -m unittest -v
```

Tests cover traversal coverage and Hilbert adjacency, scalar-reference XOR
counts at several lags, lane packing and edge replication, and native JPEG
sampling on odd dimensions. No external image fixtures are needed.
