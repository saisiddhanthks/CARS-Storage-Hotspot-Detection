#!/usr/bin/env bash
# Robustness variants: subtle severity, partial victims, and both combined.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${CARS_OUT:-../work/out}"; mkdir -p "$OUT"
for v in subtle partial subtle_partial; do
  python3 run_baselines.py "$v" > "$OUT/bl_$v.log" 2>&1
  python3 run_cars.py "$v"      > "$OUT/cars_$v.log" 2>&1
  python3 run_nn.py "$v"        > "$OUT/nn_$v.log" 2>&1
  echo "variant $v finished"
done
