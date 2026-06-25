#!/usr/bin/env bash
set -euo pipefail

# stable-audio-tools 0.0.20 (the version with stable-audio-3 support) requires
# Python >=3.10,<3.11. Newer Python silently falls back to an older, incompatible
# release. Pin to 3.10.
PYBIN="${PYBIN:-python3.10}"

if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "ERROR: $PYBIN not found. Install it:  brew install python@3.10"
  echo "Or set PYBIN to a 3.10 interpreter:  PYBIN=/path/to/python3.10 ./setup.sh"
  exit 1
fi

"$PYBIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Second step: override sat's PyWavelets==1.4.1 pin (numpy 2 ABI fix). Resolver
# warns about the pin mismatch but installs; that warning is expected.
pip install --upgrade "PyWavelets>=1.7"

echo
echo "Done. Next:"
echo "  1. cp .env.example .env   then paste your HF_TOKEN"
echo "  2. Accept license: https://huggingface.co/stabilityai/stable-audio-3-small-music"
echo "  3. source .venv/bin/activate"
echo "  4. python generate.py \"house music festival 124 BPM\" --duration 30"
