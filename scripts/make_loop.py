#!/usr/bin/env python3
"""Turn a generated track into a seamless loop.

The model renders a track with a hard start and a hard end; played on repeat
that boundary clicks. This wraps the tail of the audio back over its head with
an equal-power crossfade so the end flows into the start with no seam.

    python scripts/make_loop.py output/fantasy/fantasy_01_*.wav
    python scripts/make_loop.py track.wav --xfade 2.0 --out track_loop.wav
    # trim to a whole number of bars first (cleanest musical loop):
    python scripts/make_loop.py track.wav --bpm 120 --bars 16 --xfade 1.0

Output length = (trimmed length) - xfade. Loop point is sample 0, so play the
result on repeat (e.g. `afplay` in a loop, or a game audio engine set to loop).
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf


def equal_power_loop(audio: np.ndarray, sr: int, xfade_s: float) -> np.ndarray:
    """Overlap the last xfade seconds onto the first xfade seconds.

    audio: (samples, channels). Returns (samples - xfade, channels) where the
    head already contains the blended tail, so result[-1] -> result[0] is
    continuous. Equal-power (sin/cos) curves keep perceived loudness constant
    through the blend instead of dipping like a linear fade."""
    n = audio.shape[0]
    x = int(round(xfade_s * sr))
    if x < 1:
        return audio
    if x * 2 > n:
        raise SystemExit(f"xfade {xfade_s}s too long for {n/sr:.1f}s audio")

    loop_len = n - x
    out = audio[:loop_len].copy().astype(np.float64)

    t = np.linspace(0.0, 1.0, x, endpoint=False)
    w_in = np.sin(t * np.pi / 2.0)[:, None]    # head rises 0 -> 1
    w_out = np.cos(t * np.pi / 2.0)[:, None]   # wrapped tail falls 1 -> 0

    head = audio[:x].astype(np.float64)
    tail = audio[loop_len:loop_len + x].astype(np.float64)
    out[:x] = head * w_in + tail * w_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Make a WAV loop seamlessly.")
    ap.add_argument("input", help="Input WAV (globs allowed, first match used).")
    ap.add_argument("--xfade", type=float, default=1.5,
                    help="Crossfade seconds at the loop seam (default 1.5).")
    ap.add_argument("--bpm", type=float, default=None,
                    help="If set with --bars, trim to a whole number of bars "
                         "before looping.")
    ap.add_argument("--bars", type=int, default=None,
                    help="Bar count to trim to (assumes 4/4). Needs --bpm.")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <input>_loop.wav).")
    args = ap.parse_args()

    matches = glob.glob(args.input)
    if not matches:
        print(f"ERROR: no file matches {args.input!r}", file=sys.stderr)
        return 1
    path = sorted(matches)[0]

    audio, sr = sf.read(path, always_2d=True)  # (samples, channels), float64

    if args.bpm and args.bars:
        bar_samples = int(round((60.0 / args.bpm) * 4 * args.bars * sr))
        # need bar_samples + xfade worth of material to wrap from
        x = int(round(args.xfade * sr))
        want = bar_samples + x
        if want > audio.shape[0]:
            print(f"WARN: {args.bars} bars @ {args.bpm} BPM ({want/sr:.1f}s) "
                  f"exceeds track ({audio.shape[0]/sr:.1f}s); not trimming.",
                  file=sys.stderr)
        else:
            audio = audio[:want]

    looped = equal_power_loop(audio, sr, args.xfade)

    peak = np.max(np.abs(looped))
    if peak > 1.0:
        looped = looped / peak  # guard against blend overshoot clipping

    out = args.out or os.path.splitext(path)[0] + "_loop.wav"
    sf.write(out, looped, sr, subtype="PCM_16")
    print(f"Saved {out}  ({looped.shape[0]/sr:.1f}s, xfade {args.xfade}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
