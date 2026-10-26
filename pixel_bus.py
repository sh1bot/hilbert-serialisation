#!/usr/bin/env python3
"""Count data-lane transitions for fixed chunks in three spatial orders."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path, mode="auto", turbojpeg_library=None):
    """Return planes, subsampling factors, and explicit decoding metadata."""
    with Image.open(path) as im:
        w, h = im.size
        if im.format == "JPEG" and mode != "rgb":
            from turbojpeg import TurboJPEG, TJCS_YCbCr, TJCS_GRAY
            decoder = TurboJPEG(turbojpeg_library)
            data = Path(path).read_bytes()
            _, _, sampling, colorspace = decoder.decode_header(data)
            factors = {0: (1, 1), 1: (2, 1), 2: (2, 2),
                       3: (1, 1), 4: (1, 2), 5: (4, 1)}
            if colorspace not in (TJCS_YCbCr, TJCS_GRAY) or sampling not in factors:
                raise ValueError("Unsupported native JPEG colorspace/sampling; use --mode rgb explicitly")
            sx, sy = factors[sampling]
            raw = decoder.decode_to_yuv_planes(data)
            names = ["Y"] if colorspace == TJCS_GRAY else ["Y", "U", "V"]
            scales = {name: ((1, 1) if name == "Y" else (sx, sy)) for name in names}
            planes = {name: p[:math.ceil(h/scales[name][1]), :math.ceil(w/scales[name][0])]
                      for name, p in zip(names, raw)}
            source = "native JPEG components (U=Cb, V=Cr); no RGB conversion"
        else:
            if mode == "native":
                raise ValueError("--mode native requires a JPEG")
            rgb = np.asarray(im.convert("RGB"))
            planes = {name: rgb[:, :, i] for i, name in enumerate("RGB")}
            scales = {name: (1, 1) for name in planes}
            source = "Pillow RGB8 decode; alpha ignored; no YUV conversion"
    return planes, scales, {"width": w, "height": h, "decode": source,
                            "subsampling_factors": scales,
                            "orientation": "stored image orientation; EXIF rotation not applied"}


def hilbert_index(x, y, side):
    d, s = 0, side // 2
    while s:
        rx, ry = bool(x & s), bool(y & s)
        d += s * s * ((3 * rx) ^ ry)
        if not ry:
            if rx:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        s //= 2
    return d


def coordinates(width, height, order):
    """Sort only real coordinates: equivalent to skipping virtual square cells."""
    coords = [(x, y) for y in range(height) for x in range(width)]
    side = 1 << (max(width, height) - 1).bit_length()
    if order == "hilbert":
        coords.sort(key=lambda p: hilbert_index(*p, side))
    elif order == "z":
        def morton(p):
            x, y = p
            return sum(((x >> b) & 1) << (2*b) | ((y >> b) & 1) << (2*b+1)
                       for b in range(side.bit_length() - 1))
        coords.sort(key=morton)
    elif order != "raster":
        raise ValueError(order)
    return np.asarray(coords, dtype=np.intp)


def pack_chunks(planes, scales, width, height, chunk):
    cw, ch = chunk
    gx, gy = math.ceil(width/cw), math.ceil(height/ch)
    packed, shapes = {}, {}
    for name, plane in planes.items():
        sx, sy = scales[name]
        if cw % sx or ch % sy:
            raise ValueError(f"Chunk {cw}x{ch} must align to {sx}x{sy} sampling for {name}")
        pw, ph = cw//sx, ch//sy
        padded = np.pad(plane, ((0, gy*ph-plane.shape[0]), (0, gx*pw-plane.shape[1])), mode="edge")
        packed[name] = padded.reshape(gy, ph, gx, pw).transpose(0, 2, 1, 3).reshape(gy*gx, ph*pw)
        shapes[name] = [ph, pw]
    return packed, shapes, gx, gy


def count_transitions(words, indices, lag, batch_size=65536):
    """Rows of result are fixed sample lanes, columns are bits 0 (LSB)..7."""
    counts = np.zeros((words.shape[1], 8), dtype=np.int64)
    for start in range(lag, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        xor = words[indices[start:end]] ^ words[indices[start-lag:end-lag]]
        for bit in range(8):
            counts[:, bit] += ((xor >> bit) & 1).sum(axis=0, dtype=np.int64)
    comparisons = max(0, len(indices)-lag)
    total = int(counts.sum())
    return {"total": total, "by_bit_lsb_first": counts.sum(axis=0).tolist(),
            "by_lane_lsb_first": counts.tolist(),
            "transition_fraction": total/(comparisons*words.shape[1]*8) if comparisons else None}


def analyze(planes, scales, metadata, chunks, lags):
    report = {"image": metadata, "boundary": "edge replication; no initial load or frame wraparound",
              "lane_order": "channel-local row-major sample positions, bits LSB first", "results": []}
    for chunk in chunks:
        packed, shapes, gx, gy = pack_chunks(planes, scales, metadata["width"], metadata["height"], chunk)
        for lag in lags:
            baseline = None
            for order in ("raster", "z", "hilbert"):
                coords = coordinates(gx, gy, order)
                indices = coords[:, 1]*gx + coords[:, 0]
                channels = {name: count_transitions(words, indices, lag) for name, words in packed.items()}
                total = sum(c["total"] for c in channels.values())
                if baseline is None:
                    baseline = total
                report["results"].append({"chunk": list(chunk), "grid": [gx, gy], "order": order,
                    "lag": lag, "cycles": gx*gy, "comparisons": max(0, gx*gy-lag),
                    "bus_bits": sum(p.shape[1]*8 for p in packed.values()), "channel_lane_shapes": shapes,
                    "channels": channels, "total": total,
                    "reduction_vs_raster_percent": 100*(baseline-total)/baseline if baseline else None})
    return report


def chunk_arg(value):
    try:
        w, h = map(int, value.lower().split("x"))
        if min(w, h) > 0:
            return w, h
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected positive WIDTHxHEIGHT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--chunk", type=chunk_arg, action="append", help="repeatable WIDTHxHEIGHT; default sampling minimum")
    parser.add_argument("--lag", type=int, action="append", help="repeatable positive lag; default 1")
    parser.add_argument("--mode", choices=("auto", "rgb", "native"), default="auto")
    parser.add_argument("--turbojpeg-library", help="optional path to libturbojpeg")
    parser.add_argument("--json", type=Path, help="write full per-lane results")
    args = parser.parse_args()
    if any(lag < 1 for lag in (args.lag or [1])):
        parser.error("lags must be positive")
    try:
        planes, scales, meta = load_image(args.image, args.mode, args.turbojpeg_library)
        default = (max(s[0] for s in scales.values()), max(s[1] for s in scales.values()))
        report = analyze(planes, scales, meta, args.chunk or [default], args.lag or [1])
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(meta["decode"])
    print("chunk   lag order      bits     transitions  reduction vs raster   channel totals")
    for r in report["results"]:
        reduction = r["reduction_vs_raster_percent"]
        delta = "n/a" if reduction is None else f"{reduction:.2f}%"
        channels = " ".join(f"{k}={v['total']}" for k, v in r["channels"].items())
        print(f"{r['chunk'][0]}x{r['chunk'][1]:<4} {r['lag']:>3} {r['order']:<8} {r['bus_bits']:>6} {r['total']:>15} {delta:>20}   {channels}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
