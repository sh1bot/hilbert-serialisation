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

## Tiered traversal

```sh
python pixel_bus.py photo.jpg --macroblock 64x64 --chunk 4x4 --lag 1 --json tiered.json
python pixel_bus.py photo.jpg --macroblock 64x64 --chunk 4x4 --within-chunk extend --json extended.json
```

`--macroblock WIDTHxHEIGHT` visits macroblocks in raster order. Within each
macroblock the experiment visits chunks in raster, Z, or Hilbert order, starting
the same curve orientation in every macroblock. Both macroblock dimensions must
be multiples of the corresponding chunk dimensions. Omit this option to retain
whole-image chunk traversal. Edge replication fills complete edge macroblocks;
all their chunks are transferred, including chunks entirely in the padded area.
The reported cycle count includes that padding. Transition/lag history continues
across macroblock boundaries; it is never reset there.

`--within-chunk raster` is the default: fixed row-major sample lanes inside each
chunk, regardless of its parent curve orientation.

`--within-chunk extend` continues the selected parent curve down into the chunk.
All samples still transfer simultaneously; the option changes their assignment
to bus lanes. Hilbert inherits the enclosing curve's orientation, giving four
possible lane permutations for nontrivial square power-of-two chunks. Z-order
uses a fixed local Morton permutation; raster retains row-major packing. With
all other settings fixed this gives four distinct experiments in the usual case:
raster, Z, Hilbert with raster lanes, and Hilbert with inherited lanes. Fixed
local Z packing is just a common lane permutation, so its aggregate toggle counts
are unchanged, although its per-lane labels differ. Degenerate chunks may make
some of these cases identical.

Extended packing currently requires square power-of-two chunks and square
power-of-two macroblocks. Without an explicit macroblock, the chunk-padded image
must form a square power-of-two region. Rectangular geometry is supported in the
default raster packing mode. Each subsampled channel orders its own samples by
their full-resolution positions on the parent curve, with no channel mixing.
In extended mode, JSON lane numbers identify positions in the inherited scan,
rather than fixed spatial positions. Bits within each sample remain LSB first.

The tests also verify raster macroblock ordering, continuous comparisons across
boundaries, the four Hilbert orientations, and agreement between the flattened
hierarchical curve and a full-resolution curve.

## Delta encoding at the FIFO input

Every configuration now produces paired `raw` and `delta-sm` rows automatically.
`vs raw` gives the percentage toggle reduction from enabling the encoder for
that exact traversal, chunk packing and lag. Negative values mean an increase.
`vs raster` compares against raster with the **same encoding**. JSON also gives
per-channel `reduction_vs_raw_percent`; zero denominators produce null.

The predictor stores the previous complete transfer, independently for each
8-bit sample lane, after any within-chunk permutation. It resets every lane to
zero at each macroblock. Without `--macroblock`, the entire image is one predictor
reset region. Predictor distance is always one cycle, independently of FIFO lag.
There is no prediction between different lanes or channels.

Compute `d = (current - previous) mod 256`. Codes are:

| Modular delta | Code |
| --- | --- |
| 0..127 | 0..127 (positive sign, seven-bit magnitude) |
| 129..255 | `0x80 OR (256-d)` (negative sign, seven-bit magnitude) |
| 128 | `0x80` (negative-zero code repurposed as the half-range delta) |

The decoder maps `0x80` to a delta of 128, otherwise applies the indicated sign
to the magnitude, adds to its reconstructed previous lane value modulo 256, and
resets at the corresponding macroblock boundary. No extra data bit is needed.
In the encoded rows bit 7 is the sign/escape bit, not the original sample MSB.

The first transfer in a macroblock is encoded against zero; subsequent transfers
are encoded against their predecessor. FIFO transition comparisons remain
continuous across macroblocks, including encoded boundary transfers. As before,
there is no comparison against an initial FIFO value. Decoder resets must track
macroblock boundaries in the delayed output stream. Scores cover encoded FIFO
data only; predictor flops, encoder/decoder logic and boundary signalling are
excluded. A reference decoder is included and tested exhaustively over all
65,536 previous/current byte pairs, with additional multi-cycle reset tests.
