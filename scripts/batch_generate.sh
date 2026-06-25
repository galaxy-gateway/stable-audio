#!/usr/bin/env bash
# Batch-generate N audio tracks from a list of prompts, one per line.
#
# Usage:
#   scripts/batch_generate.sh <name> <prompts_file> [DUR] [STEPS] [SEED_BASE]
#
#   <name>          label for the output subdir: output/<name>/<name>_NN.wav
#   <prompts_file>  text file, one prompt per line (blank lines skipped)
#   DUR             seconds per track   (default 60)
#   STEPS           diffusion steps     (default 8)
#   SEED_BASE       seed for track i = SEED_BASE + i  (default 100)
#
# Example:
#   scripts/batch_generate.sh melodic_metal prompts/melodic_metal.txt 60
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

NAME=${1:?usage: batch_generate.sh <name> <prompts_file> [DUR] [STEPS] [SEED_BASE]}
PROMPTS_FILE=${2:?missing prompts_file}
DUR=${3:-60}
STEPS=${4:-8}
SEED_BASE=${5:-100}

OUTDIR="output/$NAME"
mkdir -p "$OUTDIR"

i=0
while IFS= read -r p || [ -n "$p" ]; do
  [ -z "${p// }" ] && continue          # skip blank lines
  case "$p" in \#*) continue ;; esac      # skip comment lines
  i=$((i+1))
  out=$(printf "%s/%s_%02d.wav" "$OUTDIR" "$NAME" "$i")
  echo "=== [$i] $out ==="
  python generate.py "$p" --duration "$DUR" --steps "$STEPS" \
    --seed "$((SEED_BASE+i))" --out "$out"
done < "$PROMPTS_FILE"

echo "DONE. $i files in $OUTDIR/"
ls -la "$OUTDIR/"
