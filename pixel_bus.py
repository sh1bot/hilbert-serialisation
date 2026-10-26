#!/usr/bin/env python3
"""Count data-lane transitions for chunk traversals, optionally within raster macroblocks."""
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


def transfer_layout(gx, gy, order, chunk, macroblock=None):
    """Raster macroblocks, traversal-local chunks; never reset the lag history."""
    cw, ch = chunk
    if macroblock is None:
        return coordinates(gx, gy, order), (gx, gy)
    mw, mh = macroblock
    if mw % cw or mh % ch:
        raise ValueError("Macroblock dimensions must be multiples of chunk dimensions")
    mx, my = mw // cw, mh // ch
    local = coordinates(mx, my, order)
    coords = [(bx+x, by+y) for by in range(0, gy, my)
              for bx in range(0, gx, mx) for x, y in local]
    return np.asarray(coords, dtype=np.intp), (mx, my)


def extend_lanes(words, shape, coords, grid_width, local_grid, chunk, scale, order):
    """Continue the parent curve down to sample lanes, inheriting its orientation."""
    ph, pw = shape
    cw, ch = chunk
    sx, sy = scale
    mx, my = local_grid
    side = mx*cw
    if cw != ch or side != my*ch or side & (side-1) or cw & (cw-1):
        raise ValueError("Extended packing requires square power-of-two chunks and macroblocks "
                         "(or a square power-of-two padded image when no macroblock is given)")
    result = np.empty_like(words)
    permutations = {}
    for x, y in coords:
        lx, ly = int(x % mx), int(y % my)
        key = (lx, ly)
        if key not in permutations:
            samples = [(u, v) for v in range(ph) for u in range(pw)]
            if order == "hilbert":
                samples.sort(key=lambda uv: hilbert_index(lx*cw+uv[0]*sx, ly*ch+uv[1]*sy, side))
            elif order == "z":
                # Morton ordering has the same orientation in every child.
                samples = list(map(tuple, coordinates(pw, ph, "z")))
            permutations[key] = [v*pw+u for u, v in samples]
        index = y*grid_width+x
        result[index] = words[index, permutations[key]]
    return result


def fold_signed_bytes(values):
    """Complement low seven bits of negative uint8 codes; self-inverse."""
    values = np.asarray(values, dtype=np.uint8)
    return values ^ np.where(values & 0x80, 0x7f, 0).astype(np.uint8)


def fold_biased_chroma(values):
    """Center unsigned chroma at 128, then fold its signed byte representation."""
    return fold_signed_bytes(np.asarray(values, dtype=np.uint8) ^ np.uint8(0x80))


def analyze(planes, scales, metadata, chunks, lags, macroblock=None, within_chunk="raster"):
    report = {"image": metadata, "boundary": "edge replication to complete macroblocks/chunks; continuous lag history; no initial load or frame wraparound",
              "lane_order": within_chunk, "macroblock": macroblock,
              "macroblock_order": "raster", "results": []}
    for chunk in chunks:
        width, height = metadata["width"], metadata["height"]
        if macroblock:
            if macroblock[0] % chunk[0] or macroblock[1] % chunk[1]:
                raise ValueError("Macroblock dimensions must be multiples of chunk dimensions")
            width = math.ceil(width/macroblock[0])*macroblock[0]
            height = math.ceil(height/macroblock[1])*macroblock[1]
        packed, shapes, gx, gy = pack_chunks(planes, scales, width, height, chunk)
        for lag in lags:
            baseline = None
            baseline_channels = None
            for order in ("raster", "z", "hilbert"):
                coords, local_grid = transfer_layout(gx, gy, order, chunk, macroblock)
                indices = coords[:, 1]*gx + coords[:, 0]
                streams = {}
                for name, words in packed.items():
                    if within_chunk == "extend":
                        words = extend_lanes(words, shapes[name], coords, gx, local_grid,
                                             chunk, scales[name], order)
                    streams[name] = words[indices]
                sequential = np.arange(len(indices))
                folded_channels = {name: count_transitions(fold_signed_bytes(stream), sequential, lag)
                                   for name, stream in streams.items()}
                folded_total = sum(c["total"] for c in folded_channels.values())
                uv_channels = ({name: (folded_channels[name].copy() if name in ("U", "V") else
                                       count_transitions(stream, sequential, lag))
                                for name, stream in streams.items()}
                               if "U" in streams and "V" in streams else None)
                uv_total = sum(c["total"] for c in uv_channels.values()) if uv_channels else None
                channels = {}
                for name, stream in streams.items():
                    channels[name] = count_transitions(stream, sequential, lag)
                total = sum(c["total"] for c in channels.values())
                if baseline is None:
                    baseline = total
                    baseline_channels = {name: channel["total"] for name, channel in channels.items()}
                for name, channel in channels.items():
                    raw = baseline_channels[name]
                    channel["baseline_raw_raster_total"] = raw
                    channel["reduction_vs_raw_percent"] = 100*(raw-channel["total"])/raw if raw else None
                for name, channel in folded_channels.items():
                    raw = baseline_channels[name]
                    channel["reduction_vs_raw_percent"] = 100*(raw-channel["total"])/raw if raw else None
                if uv_channels:
                    for name, channel in uv_channels.items():
                        raw = baseline_channels[name]
                        channel["reduction_vs_raw_percent"] = 100*(raw-channel["total"])/raw if raw else None
                report["results"].append({"chunk": list(chunk), "grid": [gx, gy], "order": order,
                        "fold_uv_only": ({"total": uv_total, "channels": uv_channels,
                                      "reduction_vs_raw_percent": 100*(baseline-uv_total)/baseline if baseline else None}
                                     if uv_channels else None),
                    "fold_only": {"total": folded_total, "channels": folded_channels,
                                  "reduction_vs_raw_percent": 100*(baseline-folded_total)/baseline if baseline else None},
                    "within_chunk": within_chunk, "macroblock": macroblock,
                    "lag": lag, "cycles": gx*gy, "comparisons": max(0, gx*gy-lag),
                    "bus_bits": sum(p.shape[1]*8 for p in packed.values()), "channel_lane_shapes": shapes,
                    "channels": channels, "total": total,
                    "baseline_raw_raster_total": baseline,
                    "reduction_vs_raw_percent": 100*(baseline-total)/baseline if baseline else None,
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


def format_summary(report):
    """One row per order, all savings relative to unencoded raster."""
    lines = ["macroblock chunk   lag order      bits     transitions    vs raster     fold bit 8  fold UV bit 8"]
    baseline = None
    for raw in report["results"]:
        if raw["order"] == "raster":
            baseline = raw["total"]
        def reduction(total):
            return f"{100*(baseline-total)/baseline:.2f}%" if baseline else "n/a"
        uv = raw.get("fold_uv_only")
        uv_reduction = reduction(uv["total"]) if uv else "n/a"
        macroblock = raw.get("macroblock")
        macro_label = f"{macroblock[0]}x{macroblock[1]}" if macroblock else "image"
        lines.append(f"{macro_label:<10} {raw['chunk'][0]}x{raw['chunk'][1]:<4} {raw['lag']:>3} "
                     f"{raw['order']:<8} {raw['bus_bits']:>6} "
                     f"{raw['total']:>15} {reduction(raw['total']):>12} {reduction(raw['fold_only']['total']):>14} {uv_reduction:>14}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--chunk", type=chunk_arg, action="append", help="repeatable WIDTHxHEIGHT; default sampling minimum")
    parser.add_argument("--macroblock", type=chunk_arg, action="append",
                        help="repeatable WIDTHxHEIGHT; macroblocks visited in raster order")
    parser.add_argument("--within-chunk", choices=("raster", "extend"), default="raster",
                        help="fixed raster lanes or continue traversal into chunk sample lanes")
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
        reports = [analyze(planes, scales, meta, args.chunk or [default], args.lag or [1],
                           macroblock, args.within_chunk)
                   for macroblock in (args.macroblock or [None])]
        report = reports[0]
        report.pop("macroblock")
        report["macroblocks"] = args.macroblock or [None]
        report["results"] = [row for item in reports for row in item["results"]]
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(meta["decode"])
    print(format_summary(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
