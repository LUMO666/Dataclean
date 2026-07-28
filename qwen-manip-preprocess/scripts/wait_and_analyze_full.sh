#!/usr/bin/env bash
# Monitor full humanoid run and generate analysis report when finished.
OUT="/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged_qwenmanip_processed"
ROOT="/mnt/project_rlinf_hs/liuweilin/Dataclean/qwen-manip-preprocess"

while pgrep -f "run_humanoid_full.py.*humanoid_merged_qwenmanip_processed" >/dev/null; do
  sleep 300
done

cd "$ROOT" || exit 1
PYTHONPATH=src python3 scripts/analyze_full_run.py --output-dir "$OUT"
echo "Analysis written to $OUT/analysis_report.json and $OUT/analysis_report.md"
