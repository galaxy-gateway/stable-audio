#!/usr/bin/env python3
"""Reshape stable-audio clips into usable album transitions/segues.

The model composes a self-contained piece: energy peaks mid-clip (~3-5s) and
decays to silence by ~9s inside its fixed ~16s window. That tail is dead air,
so raw clips do not flow into the next song. This tool detects the real content
region and reshapes it per the shape token in the filename so each clip ends
with intent:

  riser   reverse the content -> quiet builds to a peak at the very end
          (momentum slams into the next track). Tiny fade-in, hard end.
  downer  keep the natural loud->quiet decay (already a wind-down); trim the
          trailing silence. Short fade-out resolve.
  swell   center a symmetric window on the energy peak; fade both edges.
  drone   tile a steady mid segment with crossfades to sustain to length;
          fade both edges. (model cannot hold a note, so we build the hold.)
  weird   keep content, light edge fades.

Usage: scripts/segue_shape.py [INDIR] [OUTDIR] [--len SECONDS]
  INDIR  default output/transitions
  OUTDIR default <INDIR>/segue
  --len  target length cap in seconds (default 8). Clips are never padded with
         silence; this only caps. Content shorter than --len stays its length.
"""
import sys
import os
import glob
import argparse
import numpy as np
import soundfile as sf

CONTENT_DB = -32.0  # signal below (peak + this) is treated as silence/decay


def shape_of(name):
    n = name.lower()
    for s in ("riser", "swell", "drone", "downer", "weird"):
        if s in n:
            return s
    return "other"


def rms_envelope(x, sr, win=0.02):
    """Per-window RMS in dB, plus the window length in samples."""
    n = max(1, int(sr * win))
    mono = x.mean(axis=1) if x.ndim > 1 else x
    pad = (-len(mono)) % n
    if pad:
        mono = np.concatenate([mono, np.zeros(pad)])
    frames = mono.reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return db, n


def content_bounds(x, sr):
    """Sample indices [start, end) of the real content region."""
    db, n = rms_envelope(x, sr)
    thresh = db.max() + CONTENT_DB
    loud = np.where(db > thresh)[0]
    if len(loud) == 0:
        return 0, len(x)
    start = loud[0] * n
    end = min(len(x), (loud[-1] + 1) * n)
    return int(start), int(end)


def peak_index(x, sr):
    db, n = rms_envelope(x, sr)
    return int(db.argmax() * n)


def fade(seg, sr, t_in, t_out):
    seg = seg.copy()
    ni = min(len(seg), int(sr * t_in))
    no = min(len(seg), int(sr * t_out))
    if ni > 0:
        ramp = np.linspace(0, 1, ni)
        seg[:ni] *= ramp[:, None] if seg.ndim > 1 else ramp
    if no > 0:
        ramp = np.linspace(1, 0, no)
        seg[-no:] *= ramp[:, None] if seg.ndim > 1 else ramp
    return seg


def xfade_tile(seg, sr, target_len, xf=0.4):
    """Repeat seg to reach target_len, crossfading each splice for a smooth
    sustained bed (used for drones the model can't hold)."""
    nxf = int(sr * xf)
    if len(seg) <= nxf * 2:
        return seg
    out = seg.copy()
    while len(out) < target_len:
        a = out[-nxf:]
        b = seg[:nxf]
        ramp = np.linspace(1, 0, nxf)
        ramp = ramp[:, None] if seg.ndim > 1 else ramp
        cross = a * ramp + b * (1 - ramp)
        out = np.concatenate([out[:-nxf], cross, seg[nxf:]])
    return out[:target_len]


def process(x, sr, shape, target_len_s):
    s, e = content_bounds(x, sr)
    body = x[s:e]
    tgt = int(sr * target_len_s)

    if shape == "riser":
        # the model swells up then decays; a riser must END at the climax, so
        # take the rising half only (content_start -> energy peak) and cut on
        # the peak. quiet builds to loud and slams straight into the next track.
        pk = peak_index(x, sr) - s
        pk = max(int(sr * 1.0), pk)     # guard against a too-early peak
        rise = body[:pk]
        if len(rise) > tgt:
            rise = rise[-tgt:]
        # force a real crescendo: the model only climbs a few dB, so multiply by
        # a curve from ~ -14 dB up to 0 dB so the end clearly dominates.
        g = np.linspace(0.2, 1.0, len(rise)) ** 1.5
        rise = rise * (g[:, None] if rise.ndim > 1 else g)
        out = fade(rise, sr, 0.2, 0.012)  # tiny end guard, no real fade-out

    elif shape == "downer":
        if len(body) > tgt:
            body = body[:tgt]
        out = fade(body, sr, 0.05, 0.6)   # let it resolve down

    elif shape == "swell":
        pk = peak_index(x, sr) - s
        half = tgt // 2
        a = max(0, pk - half)
        b = min(len(body), a + tgt)
        a = max(0, b - tgt)
        seg = body[a:b]
        out = fade(seg, sr, min(1.5, target_len_s / 3), min(1.8, target_len_s / 3))

    elif shape == "drone":
        # use a steady chunk around the peak, tile to length for true sustain
        pk = peak_index(x, sr) - s
        grab = int(sr * min(3.0, len(body) / sr))
        a = max(0, min(pk - grab // 2, len(body) - grab))
        seg = body[a:a + grab]
        held = xfade_tile(seg, sr, tgt)
        out = fade(held, sr, 0.8, 1.2)

    else:  # weird / other
        if len(body) > tgt:
            body = body[:tgt]
        out = fade(body, sr, 0.2, 0.4)

    # normalize to -1 dBFS peak
    peak = np.max(np.abs(out)) + 1e-9
    out = out / peak * 0.89
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", nargs="?", default="output/transitions")
    ap.add_argument("outdir", nargs="?", default=None)
    ap.add_argument("--len", type=float, default=8.0, dest="length")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.indir, "segue")
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.indir, "*.wav")))
    if not files:
        print(f"No WAVs in {args.indir}", file=sys.stderr)
        sys.exit(1)

    for f in files:
        base = os.path.basename(f)
        x, sr = sf.read(f, always_2d=True)
        shape = shape_of(base)
        out = process(x, sr, shape, args.length)
        dst = os.path.join(outdir, base)
        sf.write(dst, out, sr, subtype="PCM_16")
        print(f"shaped [{shape:6}] {base}  {len(out)/sr:.2f}s")

    print(f"DONE -> {outdir}")


if __name__ == "__main__":
    main()
