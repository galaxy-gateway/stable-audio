#!/usr/bin/env python3
"""Generate music with stabilityai/stable-audio-3-small-music.

Usage:
    python generate.py "lofi hip hop beat to study to 85 BPM"
    python generate.py "epic orchestral trailer 140 BPM" --duration 60 --out track.wav
    python generate.py --prompt-file song.txt        # paste-friendly, no escaping
    pbpaste | python generate.py                      # pipe from clipboard
"""
import argparse
import os
import re
import sys
import unicodedata

import soundfile as sf
import torch
from dotenv import load_dotenv
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

MODEL_ID = "stabilityai/stable-audio-3-small-music"

# Typographic characters that paste in from word processors / lyric sheets and
# either break shells or confuse the text encoder. Mapped to plain ASCII.
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',  # double quotes
    "–": "-", "—": "-", "―": "-", "−": "-",  # en/em/minus
    "…": "...",                                             # ellipsis
    " ": " ", " ": " ", " ": " ", " ": " ",  # fancy spaces
    "´": "'", "`": "",                                      # acute/backtick
}
# Strip entirely: C0 controls (except \t \n), zero-width, bidi, and BOM chars.
_STRIP_RE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f"      # C0 controls except tab/newline
    "​-‏"               # zero-width space .. RTL mark
    "  "               # line / paragraph separators
    "‪-‮"               # bidi embedding / override
    "⁠﻿]"               # word-joiner, BOM / zero-width nbsp
)


def sanitize_prompt(text: str) -> str:
    """Clean a pasted prompt: normalize unicode, map fancy punctuation to ASCII,
    drop control/zero-width chars, collapse blank lines and trailing space.

    Keeps structural punctuation (brackets, parens, newlines) so lyric/chord
    sheets survive intact; only removes what breaks shells or the encoder."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_CHAR_MAP.get(ch, ch) for ch in text)
    text = _STRIP_RE.sub("", text)
    # Trim trailing whitespace per line; collapse 3+ blank lines to one.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_prompt(args) -> str:
    """Resolve the prompt from --prompt-file, a positional arg, or stdin."""
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            return fh.read()
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():  # piped / heredoc input
        return sys.stdin.read()
    return ""


def loop_heal(model, audio, sample_rate, sample_size, conditioning, *,
              steps, cfg_scale, device, heal_seconds):
    """Make `audio` (channels, samples) loop seamlessly via inpainting.

    The diffusion seam is at the boundary audio[-1] -> audio[0]. We rotate the
    clip by half its length so that boundary lands in the middle, inpaint a
    short window across it (the model regenerates a smooth transition from the
    surrounding context), then rotate back. The healed seam is now at sample 0,
    so the clip loops with no discontinuity."""
    n = audio.shape[-1]
    half = n // 2
    rot = torch.cat([audio[:, half:], audio[:, :half]], dim=-1)
    seam_center = (n - half) / sample_rate          # where the old end/start now sits
    start = max(0.0, seam_center - heal_seconds / 2)
    end = seam_center + heal_seconds / 2
    print(f"Loop-heal: inpainting seam window {start:.1f}-{end:.1f}s")
    try:
        healed = generate_diffusion_cond_inpaint(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type="pingpong",
            device=device,
            inpaint_audio=(sample_rate, rot.to(device)),
            inpaint_mask_start_seconds=start,
            inpaint_mask_end_seconds=end,
        )
    except Exception as exc:  # model/API quirk - keep the un-healed clip
        print(f"WARN: loop-heal failed ({exc}); returning un-healed audio.",
              file=sys.stderr)
        return audio
    healed = rearrange(healed, "b d n -> d (b n)").to(audio.device)
    h = healed.shape[-1]
    half_h = h // 2
    # rotate back by the same half so the healed seam returns to the boundary
    return torch.cat([healed[:, half_h:], healed[:, :half_h]], dim=-1)


def trim_quiet_tail(audio, sample_rate, *, thresh_db=-32.0,
                    keep_seconds=0.4, min_seconds=8.0):
    """Cut trailing near-silence so the track ends tight.

    The model renders into a fixed 120s buffer; with a shorter `seconds_total`
    it packs the content early then decays into a long quiet reverb tail. We
    find the last 50ms window whose RMS rises above `thresh_db` below the track
    peak, keep `keep_seconds` of natural decay after it, and drop the rest.
    `min_seconds` guards against over-trimming a genuinely sparse track.

    audio: (channels, samples) float tensor. Returns the trimmed tensor."""
    n = audio.shape[-1]
    mono = audio.abs().mean(dim=0)                  # (samples,)
    win = max(1, int(0.05 * sample_rate))
    frames = mono[: (n // win) * win].unfold(0, win, win)  # (nframes, win)
    rms = frames.pow(2).mean(dim=1).sqrt()
    peak = rms.max()
    if peak <= 0:
        return audio
    thresh = peak * (10.0 ** (thresh_db / 20.0))
    above = (rms > thresh).nonzero()
    if len(above) == 0:
        return audio
    last = int(above[-1].item())
    end = (last + 1) * win + int(keep_seconds * sample_rate)
    end = max(end, int(min_seconds * sample_rate))
    end = min(n, end)
    if end < n:
        print(f"Trim-tail: {n / sample_rate:.1f}s -> {end / sample_rate:.1f}s "
              f"(cut {(n - end) / sample_rate:.1f}s of quiet)")
    return audio[:, :end]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def main() -> int:
    load_dotenv()
    if not os.getenv("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set. Copy .env.example to .env and fill it in.",
              file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Generate music from a text prompt.")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="Text prompt. Include genre, mood, BPM. "
                         "Omit to read from --prompt-file or stdin.")
    ap.add_argument("--prompt-file", help="Read the prompt from a file (UTF-8). "
                    "Best for multiline lyric/chord sheets - no shell escaping.")
    ap.add_argument("--raw", action="store_true",
                    help="Skip prompt sanitization (use the text verbatim).")
    ap.add_argument("--duration", type=int, default=120,
                    help="Length in seconds (default 120).")
    ap.add_argument("--steps", type=int, default=8,
                    help="Diffusion steps (default 8).")
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="Classifier-free guidance scale (default 1.0).")
    ap.add_argument("--out", default="outputs/output.wav", help="Output WAV path.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    ap.add_argument("--trim-tail", action="store_true",
                    help="Cut the trailing quiet reverb tail so the track ends "
                         "tight (the model pads short clips with a long decay).")
    ap.add_argument("--trim-db", type=float, default=-32.0,
                    help="Tail-trim threshold in dB below peak (default -32).")
    ap.add_argument("--loop", action="store_true",
                    help="Generation-side seamless loop: after generating, rotate "
                         "the track so its end/start seam sits mid-clip, inpaint-heal "
                         "that window, then rotate back so it loops without a seam.")
    ap.add_argument("--loop-heal-seconds", type=float, default=4.0,
                    help="Width of the seam window to regenerate when --loop (default 4).")
    args = ap.parse_args()

    prompt = read_prompt(args)
    if not prompt.strip():
        print("ERROR: no prompt. Pass one as an argument, via --prompt-file, "
              "or pipe it on stdin.", file=sys.stderr)
        return 1
    if not args.raw:
        prompt = sanitize_prompt(prompt)
    args.prompt = prompt

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} (first run downloads several GB)...")
    model, model_config = get_pretrained_model(MODEL_ID)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)
    # float16 only safe/fast on CUDA; keep float32 on mps/cpu.
    if device == "cuda":
        model = model.to(torch.float16)

    conditioning = [{"prompt": args.prompt, "seconds_total": args.duration}]

    print(f"Generating {args.duration}s: {args.prompt!r}")
    output = generate_diffusion_cond_inpaint(
        model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sampler_type="pingpong",
        device=device,
    )

    output = rearrange(output, "b d n -> d (b n)")  # (channels, samples) float

    # Trim before loop-heal: a tight end means the seam joins content->content,
    # not content->silence.
    if args.trim_tail:
        output = trim_quiet_tail(output, sample_rate, thresh_db=args.trim_db)

    if args.loop:
        output = loop_heal(
            model, output, sample_rate, sample_size, conditioning,
            steps=args.steps, cfg_scale=args.cfg_scale, device=device,
            heal_seconds=args.loop_heal_seconds,
        )

    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # torchaudio 2.7 ships no write backend on macOS; use soundfile.
    # output is (channels, samples); soundfile wants (samples, channels).
    sf.write(args.out, output.numpy().T, sample_rate, subtype="PCM_16")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
