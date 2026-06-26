---
name: batch-generate-audio
description: Generate N audio tracks in a given genre/style with stable-audio. Use when the user asks for multiple tracks, a batch, "10 files", or several variations of a genre, mood, or style. Writes one prompt per line, runs the batch detached, monitors progress.
---

# Batch-generate audio tracks

Generate any number of audio files in a genre/style using the local
`generate.py` (stabilityai/stable-audio-3-small-music). One distinct prompt per
track so the set has variety, not N copies.

## Steps

1. **Write a prompts file** — one prompt per line in `prompts/<name>.txt`
   (`<name>` = kebab/snake genre label, e.g. `melodic_metal`, `lofi_chill`).
   Each prompt = distinct subgenre/mood/instrumentation + a BPM. Blank lines and
   `#` comment lines are skipped. Make every line different — vary subgenre,
   tempo, key, instrumentation, energy. See `prompts/melodic_metal.txt` for the
   shape.

2. **Run the batch detached** (MPS is slow; ~1-3 min/track):
   ```bash
   nohup scripts/batch_generate.sh <name> prompts/<name>.txt 60 > output/<name>_batch.log 2>&1 &
   ```
   Args: `<name> <prompts_file> [DUR=60] [STEPS=8] [SEED_BASE=100]`.
   Output → `output/<name>/<name>_NN_<slug>.wav`. The `<slug>` names the
   musical idea: `batch_generate.sh` auto-derives it from each prompt's lead
   phrase (everything before the first comma), lowercased and hyphenated — so
   write that lead phrase as the track's identity (e.g. `epic fantasy
   orchestral, ...`, `haunted music box, ...`). `NN` preserves order and seed
   mapping. Seed for track i = `SEED_BASE+i` (reproducible). Re-run with a
   different `SEED_BASE` for fresh variations of the same prompts.

3. **Monitor progress** with the Monitor tool — emit per saved track, stop on
   done/error. Filter must cover failures, not just `Saved`:
   ```bash
   prev=0
   while true; do
     cur=$(grep -c 'Saved ' output/<name>_batch.log 2>/dev/null || echo 0)
     [ "$cur" -gt "$prev" ] && { echo "saved: $cur"; prev=$cur; }
     grep -q 'DONE' output/<name>_batch.log && { echo "DONE $cur"; break; }
     grep -qE 'Traceback|Error|ERROR' output/<name>_batch.log && { echo "ERROR"; grep -E 'Traceback|Error' output/<name>_batch.log | tail -3; break; }
     sleep 15
   done
   ```

4. **Verify and report** — `ls output/<name>/*.wav | wc -l` matches requested
   count. Give the user a table (track / subgenre / seed) and the preview
   command (filename now includes the slug, so glob it):
   `make play OUT="$(ls output/<name>/<name>_01_*.wav)"`.

## Notes

- Prereqs already met in this repo: `.venv`, `HF_TOKEN` in `.env`, model cached.
- Tune per-track length with the DUR arg (default 60s). `STEPS=8` is the model's
  tuned default; raising it rarely helps and costs time.
- Don't block the main thread polling — launch detached, arm Monitor, keep working.
