#!/usr/bin/env python3
"""Batch-loop every WAV in a directory (post-process crossfade).

Wraps each track's tail over its head with an equal-power crossfade so it loops
seamlessly. Optionally trims each track to a whole number of bars first, reading
per-track BPM from the prompts file used to generate it (matched by filename slug).

    python scripts/loop_dir.py output/fantasy
    python scripts/loop_dir.py output/fantasy --xfade 2.0
    # bar-aligned: pull BPM from each prompt's trailing "NNN BPM"
    python scripts/loop_dir.py output/fantasy --bars 16 --prompts prompts/fantasy.txt

Writes <name>_loop.wav next to each source; skips files already ending _loop.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf

# reuse the core crossfade from make_loop
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_loop import equal_power_loop  # noqa: E402


def slugify(text: str) -> str:
    """Same rule as batch_generate.sh: lead phrase before first comma."""
    lead = text.split(",")[0]
    return re.sub(r"-+$", "", re.sub(r"^-+", "",
                  re.sub(r"[^a-z0-9]+", "-", lead.lower())))


def load_bpm_map(prompts_file: str) -> dict:
    """slug -> BPM, parsed from each prompt line's trailing 'NNN BPM'."""
    out = {}
    with open(prompts_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.search(r"(\d+(?:\.\d+)?)\s*BPM", line, re.IGNORECASE)
            if m:
                out[slugify(line)] = float(m.group(1))
    return out


def file_slug(path: str) -> str:
    """Recover the slug from <name>_NN_<slug>.wav."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r".+?_\d+_(.+)$", base)
    return m.group(1) if m else base


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch crossfade-loop a folder of WAVs.")
    ap.add_argument("dir", help="Directory of WAV files.")
    ap.add_argument("--xfade", type=float, default=1.5, help="Crossfade seconds (default 1.5).")
    ap.add_argument("--bars", type=int, default=None, help="Trim to N bars (4/4). Needs --prompts.")
    ap.add_argument("--auto-bars", action="store_true",
                    help="Trim to the largest whole bar count that fits each track "
                         "(per-file, from its BPM and length). Needs --prompts.")
    ap.add_argument("--prompts", default=None, help="Prompts file to read per-track BPM from.")
    args = ap.parse_args()

    use_bars = (args.bars or args.auto_bars)
    bpm_map = load_bpm_map(args.prompts) if (use_bars and args.prompts) else {}

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "*.wav"))
                   if not f.endswith("_loop.wav"))
    if not files:
        print(f"No WAVs in {args.dir}", file=sys.stderr)
        return 1

    done = 0
    for path in files:
        audio, sr = sf.read(path, always_2d=True)
        x = int(round(args.xfade * sr))

        if use_bars:
            bpm = bpm_map.get(file_slug(path))
            if bpm:
                bar_sec = (60.0 / bpm) * 4
                if args.auto_bars:
                    # largest whole bars whose length + crossfade fits the track
                    bars = int((audio.shape[0] / sr - args.xfade) // bar_sec)
                    bars = max(1, bars)
                else:
                    bars = args.bars
                bar_samples = int(round(bar_sec * bars * sr))
                want = bar_samples + x
                if want <= audio.shape[0]:
                    audio = audio[:want]
                else:
                    print(f"  skip-trim {os.path.basename(path)}: "
                          f"{bars} bars @ {bpm} BPM > track length")
            else:
                print(f"  no BPM for {os.path.basename(path)}; crossfade only")

        looped = equal_power_loop(audio, sr, args.xfade)
        peak = np.max(np.abs(looped))
        if peak > 1.0:
            looped = looped / peak
        out = os.path.splitext(path)[0] + "_loop.wav"
        sf.write(out, looped, sr, subtype="PCM_16")
        done += 1
    print(f"DONE. Wrote {done} *_loop.wav in {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
