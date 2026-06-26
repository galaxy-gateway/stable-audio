# stable-audio

Generate music with [`stabilityai/stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music).
Runs locally on Apple Silicon (MPS), CUDA, or CPU.

## Prerequisites

1. **Accept the license** (required, model is gated):
   https://huggingface.co/stabilityai/stable-audio-3-small-music
2. **HF access token** with read scope: https://huggingface.co/settings/tokens
3. **Python 3.11 or 3.12.** Your system Python is 3.14, which `torch` /
   `stable-audio-tools` do not yet ship wheels for. Install one:
   ```
   brew install python@3.11
   ```

## Setup

```bash
make setup                 # venv (python3.10) + deps + scaffolds .env
# paste your HF_TOKEN into .env
make check                 # verify token, torch, MPS
```

`make` (no args) lists every target. `./setup.sh` still works if you prefer.

## Generate

```bash
make gen PROMPT="lofi hip hop beat to study to 85 BPM" DURATION=60
make gen PROMPT="deep house, warm pads, 124 BPM" SEED=42 OUT=outputs/house.wav
make play OUT=outputs/house.wav        # afplay the result
```

Vars: `PROMPT PROMPT_FILE DURATION STEPS CFG OUT SEED`. Or call the script directly:

```bash
source .venv/bin/activate
python generate.py "epic orchestral trailer 140 BPM" --duration 60 --out outputs/trailer.wav
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--prompt-file` | – | Read prompt from a UTF-8 file (no shell escaping) |
| `--raw` | off | Skip sanitization; use the text verbatim |
| `--duration` | 120 | Length in seconds |
| `--steps` | 8 | Diffusion steps (model tuned for low step counts) |
| `--cfg-scale` | 1.0 | Guidance scale |
| `--out` | outputs/output.wav | Output path |
| `--seed` | random | Reproducibility |

### Pasting messy prompts

Lyric/chord sheets full of smart quotes, em-dashes, brackets, and parens break
shell quoting. Feed them in **without escaping** instead:

```bash
make gen PROMPT_FILE=song.txt DURATION=60      # from a file
pbpaste | python generate.py                   # straight from the clipboard
python generate.py --prompt-file song.txt
```

Before generating, the prompt is **sanitized**: unicode normalized (NFKC),
smart quotes/dashes/ellipsis mapped to ASCII, backticks and zero-width/control
characters stripped, trailing whitespace and extra blank lines collapsed.
Structural punctuation (`[brackets]`, `(parens)`, newlines) is kept so lyric
sheets stay intact. Pass `--raw` to disable. The exact text used is echoed on the
`Generating ...:` line.

**Prompting tip:** include genre, mood, instrumentation, and BPM. The model is
music-specialized (no vocals/speech).

## Hardware notes

- Stability reports generation in "a few seconds on a MacBook Pro M4."
- First run downloads several GB of weights to the HF cache (`~/.cache/huggingface`).
- On Mac, the model runs in float32 (float16 is CUDA-only here).

## Alternative library

Stability also ships a simpler `stable-audio-3` package:

```python
from stable_audio_3 import StableAudioModel
model = StableAudioModel.from_pretrained("small-music")
audio = model.generate(prompt="House music ... 124 BPM", duration=120)
```

This project uses `stable-audio-tools` (the fully documented research path).
Swap in `stable-audio-3` if you prefer the higher-level API.

## Reference 

- https://github.com/Stability-AI/stable-audio-tools
- https://github.com/Stability-AI/stable-audio-metrics

## License

Model under the [Stability AI Community License](https://stability.ai/license);
commercial use requires review. Includes Google T5Gemma under Gemma Terms.
