#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT="$SCRIPT_DIR/g_00110000.pth"
CONFIG="$SCRIPT_DIR/config.yaml"
MODE="${1:-dns}"

if [[ "$MODE" == "voicebank" ]]; then
  MANIFEST="$ROOT_DIR/data/test_noisy.json"
  OUTPUT_CSV="${2:-$ROOT_DIR/results/voicebank_metrics_110000.csv}"
  AUDIO_DIR="${3:-$ROOT_DIR/results/voicebank_audio_110000}"

  echo "Running full VoiceBank evaluation"
  echo "Checkpoint : $CHECKPOINT"
  echo "Config     : $CONFIG"
  echo "Manifest   : $MANIFEST"

  mkdir -p "$(dirname "$OUTPUT_CSV")" "$AUDIO_DIR"

  conda run -n mamba python -u "$ROOT_DIR/eval_metrics.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --json_manifest "$MANIFEST" \
    --output "$OUTPUT_CSV" \
    --save_audio_dir "$AUDIO_DIR"

  echo "CSV saved to     : $OUTPUT_CSV"
  echo "Summary saved to : ${OUTPUT_CSV%.csv}_summary.txt"
  echo "Audio saved to   : $AUDIO_DIR"
  exit 0
fi

MANIFEST="$ROOT_DIR/data/eval_dns.json"
MAX_FILES="${2:-10}"
CHUNK_SECONDS="${3:-5}"
OUTPUT_CSV="${4:-$ROOT_DIR/results/dns_chunked_metrics_110000_${MAX_FILES}.csv}"
AUDIO_DIR="${5:-$ROOT_DIR/results/dns_chunked_audio_110000_${MAX_FILES}}"

echo "Running chunked DNS evaluation"
echo "Checkpoint : $CHECKPOINT"
echo "Config     : $CONFIG"
echo "Manifest   : $MANIFEST"
echo "Max files  : $MAX_FILES"
echo "Chunk sec  : $CHUNK_SECONDS"

mkdir -p "$(dirname "$OUTPUT_CSV")" "$AUDIO_DIR"

conda run -n mamba python -u "$ROOT_DIR/eval_dns_chunked_metrics.py" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --json_manifest "$MANIFEST" \
  --output "$OUTPUT_CSV" \
  --chunk_seconds "$CHUNK_SECONDS" \
  --max_files "$MAX_FILES" \
  --save_audio_dir "$AUDIO_DIR"

echo "CSV saved to     : $OUTPUT_CSV"
echo "Summary saved to : ${OUTPUT_CSV%.csv}_summary.txt"
echo "Audio saved to   : $AUDIO_DIR"
