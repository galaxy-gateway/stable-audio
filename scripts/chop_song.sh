#!/usr/bin/env bash
# Chop a song into fixed-length clips for gapless / segue testing.
#
# Usage: scripts/chop_song.sh <song> [SECONDS] [FORMAT] [OUTDIR]
#   <song>    input audio (wav/flac/mp3/...)
#   SECONDS   clip length in seconds        (default 1)
#   FORMAT    wav | flac | mp3 | opus | aac (default wav)
#   OUTDIR    output dir   (default <song-dir>/<song-name>_chops_<format>)
#
# Gapless notes:
#   wav  / flac  -> inherently gapless (lossless). Recommended for chopped albums.
#   mp3          -> encoded with `lame --nogap` so the clips form one contiguous
#                   gapless stream (NOT per-file libmp3lame, which inserts ~45ms
#                   of encoder delay/padding at every seam -> audible chop).
#   opus         -> gapless by spec (pre-skip in header); compliant players strip it.
#   aac          -> gapless only if the player honors the container delay tag.
set -euo pipefail
cd "$(dirname "$0")/.."

SONG=${1:?usage: chop_song.sh <song> [SECONDS] [FORMAT] [OUTDIR]}
SECS=${2:-1}
FMT=${3:-wav}
NAME=$(basename "${SONG%.*}")
OUTDIR=${4:-$(dirname "$SONG")/${NAME}_chops_${FMT}}

command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found"; exit 1; }
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

SR=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$SONG")
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SONG")
N=$(awk "BEGIN{print int($DUR/$SECS)}")
echo "song ${DUR}s @ ${SR}Hz -> $N clips of ${SECS}s as $FMT"

# 1) sample-accurate WAV clips into a temp dir (exact length, no codec padding)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
i=0
while [ "$i" -lt "$N" ]; do
   st=$(awk "BEGIN{print $i*$SECS}")
  ffmpeg -hide_banner -loglevel error -y -i "$SONG" -ss "$st" -t "$SECS" -c:a pcm_s16le \
    "$(printf '%s/clip_%03d.wav' "$TMP" "$i")"
  i=$((i+1))
done

# 2) encode to the requested format
case "$FMT" in
  wav)
    cp "$TMP"/clip_*.wav "$OUTDIR"/
    ;;
  flac)
    for f in "$TMP"/clip_*.wav; do
      b=$(basename "$f" .wav)
      ffmpeg -hide_banner -loglevel error -y -i "$f" -c:a flac "$OUTDIR/$b.flac"
    done
    ;;
  mp3)
    command -v lame >/dev/null || { echo "ERROR: lame CLI required for gapless mp3 (brew install lame)"; exit 1; }
    # one contiguous gapless stream; no Xing tag (it would break --nogap)
    lame --quiet --nogapout "$OUTDIR" --nogap -b 320 $(ls "$TMP"/clip_*.wav)
    ;;
  opus)
    for f in "$TMP"/clip_*.wav; do
      b=$(basename "$f" .wav)
      ffmpeg -hide_banner -loglevel error -y -i "$f" -c:a libopus -b:a 192k "$OUTDIR/$b.opus"
    done
    ;;
  aac)
    for f in "$TMP"/clip_*.wav; do
      b=$(basename "$f" .wav)
      ffmpeg -hide_banner -loglevel error -y -i "$f" -c:a aac -b:a 256k "$OUTDIR/$b.m4a"
    done
    ;;
  *)
    echo "ERROR: unknown format '$FMT' (wav|flac|mp3|opus|aac)"; exit 1 ;;
esac

echo "DONE -> $OUTDIR ($(ls "$OUTDIR" | wc -l | tr -d ' ') clips)"
