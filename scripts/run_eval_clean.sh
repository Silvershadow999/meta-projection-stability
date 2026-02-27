#!/usr/bin/env bash
set -euo pipefail

OUT="artifacts/results.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p artifacts

# Rotate old log
if [ -f "${OUT}" ]; then
  mv -v "${OUT}" "artifacts/results.jsonl.bak.${STAMP}"
fi

# Run scenarios (append into fresh OUT)
python scripts/eval_runner.py --scenario baseline --n-steps 5 --emit-every 2 --out "${OUT}"
python scripts/eval_runner.py --scenario adversarial_min --n-steps 5 --emit-every 2 --out "${OUT}"

# Validate + report
python scripts/validate_results.py --in "${OUT}" --require-scenarios baseline,adversarial_min
python scripts/eval_report.py --in "${OUT}" --out artifacts/eval_report.md

echo "OK: ${OUT}"
echo "OK: artifacts/eval_report.md"
