#!/usr/bin/env bash
# Batch-generate N audio tracks from a list of prompts, one per line.
#
# Usage:
#   scripts/batch_generate.sh <name> <prompts_file> [DUR] [STEPS] [SEED_BASE]
#
#   <name>          label for the output subdir: output/<name>/<name>_NN_<slug>.wav
#                   slug is auto-derived from the prompt's lead phrase (the
#                   musical idea, before the first comma)
#   <prompts_file>  text file, one prompt per line (blank lines skipped)
#   DUR             seconds per track   (default 60)
#   STEPS           diffusion steps     (default 8)
#   SEED_BASE       seed for track i = SEED_BASE + i  (default 100)
#   SKIP            skip the first SKIP prompts but keep counting (default 0).
#                   Use to append to an existing batch in the SAME output dir:
#                   numbering and seeds continue past the already-rendered tracks.
#
# Example:
#   scripts/batch_generate.sh melodic_metal prompts/melodic_metal.txt 60
#   # extend a finished batch: render prompts 11..N into the same dir
#   scripts/batch_generate.sh runescape_towns prompts/runescape_towns.txt 60 8 100 10
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

NAME=${1:?usage: batch_generate.sh <name> <prompts_file> [DUR] [STEPS] [SEED_BASE]}
PROMPTS_FILE=${2:?missing prompts_file}
DUR=${3:-60}
STEPS=${4:-8}
SEED_BASE=${5:-100}
SKIP=${6:-0}
# Cut the trailing quiet reverb tail so tracks end tight. On by default;
# disable with TRIM=0 scripts/batch_generate.sh ...
TRIM=${TRIM:-1}
TRIM_ARGS=()
[ "$TRIM" = "1" ] && TRIM_ARGS+=(--trim-tail)

OUTDIR="output/$NAME"
mkdir -p "$OUTDIR"

# Slug for a track name from the musical idea = the prompt's lead phrase
# (everything before the first comma), lowercased and hyphenated.
slugify() {
  echo "$1" | sed 's/,.*//' | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

i=0
while IFS= read -r p || [ -n "$p" ]; do
  [ -z "${p// }" ] && continue          # skip blank lines
  case "$p" in \#*) continue ;; esac      # skip comment lines
  i=$((i+1))
  [ "$i" -le "$SKIP" ] && continue       # already rendered; keep i/seed in sync
  slug=$(slugify "$p")
  # Name represents the musical idea: <name>_NN_<slug>.wav. NN preserves
  # order and seed mapping (seed = SEED_BASE + i); slug is human-readable.
  out=$(printf "%s/%s_%02d_%s.wav" "$OUTDIR" "$NAME" "$i" "$slug")
  echo "=== [$i] $out ==="
  python generate.py "$p" --duration "$DUR" --steps "$STEPS" \
    --seed "$((SEED_BASE+i))" --out "$out" "${TRIM_ARGS[@]}"
done < "$PROMPTS_FILE"

echo "DONE. $i files in $OUTDIR/"
ls -la "$OUTDIR/"
