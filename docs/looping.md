# Seamless looping

The model renders a track with a hard start and a hard end. Played on repeat,
that boundary clicks. Three ways to fix it, usable alone or combined.

## Methods

### 1. Crossfade wrap (post-process) — most reliable
`scripts/make_loop.py` overlaps the tail over the head with an **equal-power**
(sin/cos) crossfade. The loop point becomes sample 0 and the boundary is
continuous by construction. Output length = input − xfade.

```bash
python scripts/make_loop.py track.wav --xfade 1.5
```

### 2. Bar-aligned crossfade (post-process) — cleanest musically
Trim to a whole number of bars (4/4) before crossfading so the loop is also
rhythmically tight, not just click-free. BPM comes from the prompt's trailing
`NNN BPM`.

```bash
python scripts/make_loop.py track.wav --bpm 120 --bars 16 --xfade 1.0
```

### 3. Generation-side seam heal (`generate.py --loop`)
After generating, rotate the clip so its end→start seam lands mid-clip,
**inpaint** a short window across it (the model regenerates a smooth transition
from surrounding context), then rotate back. The healed seam is at sample 0, so
it loops with no discontinuity — and unlike crossfade it keeps full length and
can fix the *musical* phrasing across the seam, not just the amplitude.

```bash
python generate.py "synthwave loop, steady ostinato, 120 BPM" \
  --duration 30 --loop --loop-heal-seconds 4 --out track.wav
```
Costs a second diffusion pass. Falls back to the un-healed clip with a warning
if the inpaint pass errors. Best paired with a light crossfade (method 1, 0.5s).

## Batch a whole folder

```bash
python scripts/loop_dir.py output/fantasy --xfade 1.5
# bar-aligned, reading per-track BPM from the prompts file by filename slug:
python scripts/loop_dir.py output/fantasy --bars 16 --prompts prompts/fantasy.txt
```
Writes `<name>_loop.wav` next to each source; skips files already `_loop`.

## Loop-friendly prompts

- Use: `steady`, `ostinato`, `looping`, `hypnotic`, `consistent throughout`,
  `no intro`, `no fade`, `no build`.
- Avoid: `build`, `climax`, `rising`, `fade out`, `intro`, `outro`.
- Shorter clips (20–40s) loop tighter than 60s.

## Seam metric

Quick numeric check of boundary discontinuity (mean abs jump end→start):

| track            | raw seam | + crossfade |
|------------------|----------|-------------|
| plain generation | 0.0038   | 0.0 |
| gen-side heal    | 0.00002  | 0.0 |

Crossfade forces the boundary continuous; gen-side heal cuts the raw seam ~190×
on its own (useful when you want to preserve full length / no overlap).

## Test battery

`scripts/run_loop_tests.py` renders 10 musical ideas through all 7
method/combination variants (loads the model once). Ideas 1–5 at 8 steps,
6–10 at 25 steps. Output in `output/loop_tests/`, named
`<NN>_<slug>__<method>_s<steps>.wav`. Compare `m1_baseline_raw` (control) against
the others on the same idea/seed to judge each method.
