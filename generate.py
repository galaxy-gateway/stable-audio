#!/usr/bin/env python3
"""Generate music with stabilityai/stable-audio-3-small-music.

Usage:
    python generate.py "lofi hip hop beat to study to 85 BPM"
    python generate.py "epic orchestral trailer 140 BPM" --duration 60 --out track.wav
"""
import argparse
import os
import sys

import soundfile as sf
import torch
from dotenv import load_dotenv
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

MODEL_ID = "stabilityai/stable-audio-3-small-music"


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
    ap.add_argument("prompt", help="Text prompt. Include genre, mood, BPM.")
    ap.add_argument("--duration", type=int, default=120,
                    help="Length in seconds (default 120).")
    ap.add_argument("--steps", type=int, default=8,
                    help="Diffusion steps (default 8).")
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="Classifier-free guidance scale (default 1.0).")
    ap.add_argument("--out", default="outputs/output.wav", help="Output WAV path.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    args = ap.parse_args()

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

    output = rearrange(output, "b d n -> d (b n)")
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
