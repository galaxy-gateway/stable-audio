#!/usr/bin/env python3
"""Loop-method test battery.

Generates 10 musical ideas and renders each through 7 looping methods/combos so
they can be A/B compared. Loads the model ONCE (vs. re-loading per track).

Methods (output suffix):
  m1_baseline_raw      plain generation, no loop processing      (control - clicks)
  m2_crossfade         plain gen + equal-power crossfade wrap
  m3_baralign_xfade    plain gen + trim to whole bars + crossfade
  m4_loopprompt_xfade  loop-friendly prompt   + crossfade
  m5_loopprompt_bar    loop-friendly prompt   + bar-align + crossfade
  m6_inpaint_heal      gen-side seam inpaint-heal (no post)
  m7_inpaint_xfade     gen-side seam heal + light crossfade

Ideas 1-5 render at 8 steps, ideas 6-10 at 25 steps (per request).
Only 3 generations per idea (plain / loop-prompt / inpaint) feed all 7 methods.
Output: output/loop_tests/<NN>_<slug>__<method>_s<steps>.wav
"""
import os
import sys
import re

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_loop import equal_power_loop  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import loop_heal  # noqa: E402

OUTDIR = "output/loop_tests"
DURATION = 32

# (slug, bpm, generic prompt). Loop-friendly prompt is derived by appending
# steady/ostinato wording and removing build/fade cues.
IDEAS = [
    ("lofi_hiphop",      85,  "lofi hip hop beat, warm rhodes, dusty drums, mellow, 85 BPM"),
    ("synthwave_arp",    120, "synthwave, driving arpeggio, ostinato bass, neon pads, 120 BPM"),
    ("fantasy_jig",      120, "fantasy tavern jig, fiddle and lute, hand drum, lively, 120 BPM"),
    ("dungeon_synth",    100, "dungeon synth, dark medieval pads, simple repeating motif, 100 BPM"),
    ("chiptune_world",   128, "chiptune overworld, bright square lead, bouncy bass, 128 BPM"),
    ("dub_techno",       124, "dub techno, deep chord stabs, rolling sub, hypnotic, 124 BPM"),
    ("celtic_reel",      110, "celtic folk reel, tin whistle and bodhran, spirited, 110 BPM"),
    ("trap_beat",        70,  "trap beat, booming 808, crisp hats, dark keys, 70 BPM"),
    ("orchestral_ost",   120, "orchestral ostinato, staccato strings, steady pulse, 120 BPM"),
    ("ambient_pulse",    90,  "ambient electronic, warm pad swells, gentle pulse, 90 BPM"),
]

LOOP_SUFFIX = ", steady looping ostinato, consistent throughout, no intro, no fade, no build"


def gen(model, prompt, sample_rate, sample_size, steps, device, loop=False):
    cond = [{"prompt": prompt, "seconds_total": DURATION}]
    out = generate_diffusion_cond_inpaint(
        model, steps=steps, cfg_scale=1.0, conditioning=cond,
        sample_size=sample_size, sampler_type="pingpong", device=device,
    )
    audio = rearrange(out, "b d n -> d (b n)")
    if loop:
        audio = loop_heal(model, audio, sample_rate, sample_size, cond,
                          steps=steps, cfg_scale=1.0, device=device, heal_seconds=4.0)
    return audio  # (channels, samples) float tensor


def to_np(audio_t):
    a = audio_t.to(torch.float32)
    a = a / torch.max(torch.abs(a)).clamp(min=1e-8)
    return a.clamp(-1, 1).cpu().numpy().T  # (samples, channels)


def write(arr, sr, name):
    peak = np.max(np.abs(arr))
    if peak > 1.0:
        arr = arr / peak
    path = os.path.join(OUTDIR, name)
    sf.write(path, arr, sr, subtype="PCM_16")
    print(f"Saved {path}", flush=True)


def main():
    load_dotenv()
    os.makedirs(OUTDIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLoading model...", flush=True)
    model, cfg = get_pretrained_model("stabilityai/stable-audio-3-small-music")
    sr, sample_size = cfg["sample_rate"], cfg["sample_size"]
    model = model.to(device)

    for i, (slug, bpm, prompt) in enumerate(IDEAS, start=1):
        steps = 8 if i <= 5 else 25
        bars = max(1, int((DURATION - 2) // (4 * 60.0 / bpm)))  # whole bars that fit
        tag = f"{i:02d}_{slug}"
        print(f"=== idea {i}/10 {slug} (bpm {bpm}, steps {steps}, {bars} bars) ===",
              flush=True)

        # 3 base generations feed all 7 methods
        g_plain = gen(model, prompt, sr, sample_size, steps, device)
        g_loop  = gen(model, prompt + LOOP_SUFFIX, sr, sample_size, steps, device)
        g_heal  = gen(model, prompt, sr, sample_size, steps, device, loop=True)

        plain = to_np(g_plain)
        loopp = to_np(g_loop)
        heal  = to_np(g_heal)
        s = f"_s{steps}"

        # m1 baseline raw (control)
        write(plain, sr, f"{tag}__m1_baseline_raw{s}.wav")
        # m2 crossfade
        write(equal_power_loop(plain, sr, 1.5), sr, f"{tag}__m2_crossfade{s}.wav")
        # m3 bar-align + crossfade
        bar_n = int(round((60.0 / bpm) * 4 * bars * sr))
        x = int(round(1.0 * sr))
        trimmed = plain[:bar_n + x] if bar_n + x <= len(plain) else plain
        write(equal_power_loop(trimmed, sr, 1.0), sr, f"{tag}__m3_baralign_xfade{s}.wav")
        # m4 loop-prompt + crossfade
        write(equal_power_loop(loopp, sr, 1.5), sr, f"{tag}__m4_loopprompt_xfade{s}.wav")
        # m5 loop-prompt + bar-align + crossfade
        trimmed_l = loopp[:bar_n + x] if bar_n + x <= len(loopp) else loopp
        write(equal_power_loop(trimmed_l, sr, 1.0), sr, f"{tag}__m5_loopprompt_bar{s}.wav")
        # m6 inpaint heal (no post)
        write(heal, sr, f"{tag}__m6_inpaint_heal{s}.wav")
        # m7 inpaint heal + light crossfade
        write(equal_power_loop(heal, sr, 0.5), sr, f"{tag}__m7_inpaint_xfade{s}.wav")

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
