#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

out="analysis/lane_oram_security/fixed_gap5_overnight_runs"
bin="target/release/lane_oram_minibucket_security"
mkdir -p "$out"

if compgen -G "$out/rho*_rep*_ops30.csv" >/dev/null; then
  echo "refusing to overwrite existing overnight partial results" >&2
  exit 2
fi

date -u +"started=%Y-%m-%dT%H:%M:%SZ" > "$out/RUNNING"
echo "building release simulator"
"$root/.cargo/bin/cargo" build --release -p rostl-oram --bin lane_oram_minibucket_security

pids=()
labels=()
for load in 110 120; do
  blocks=$((1048576 * load / 100))
  for rep in 0 1 2 3; do
    if [ "$load" -eq 110 ]; then
      cpu=$rep
    else
      cpu=$((rep + 4))
    fi
    base="$out/rho${load}_rep${rep}_ops30"
    echo "launching load=$load replicate=$rep cpu=$cpu"
    taskset -c "$cpu" "$bin"       --n 65536       --blocks "$blocks"       --b 2       --z 8       --y 2       --queue-policy random       --access-policy selected       --access-eviction-width 1       --eviction-policy fixed       --eviction-chains 1       --path-schedule access-plus-rate-deterministic       --deterministic-numerator 1       --deterministic-denominator 5       --warmup 16777216       --operations 1073741824       --epoch-operations 67108864       --output "${base}.csv"       --epoch-output "${base}_epochs.csv"       2>"${base}.log" &
    pids+=("$!")
    labels+=("rho${load}/rep${rep}")
  done
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${labels[$index]}"
  else
    echo "failed ${labels[$index]}" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "one or more overnight simulations failed" >&2
  exit 1
fi

python3 scripts/combine_fixed_gap5_overnight.py
date -u +"completed=%Y-%m-%dT%H:%M:%SZ" > "$out/COMPLETE"
mv "$out/RUNNING" "$out/STARTED"
sha256sum "$out"/rho*_rep*_ops30.csv "$out"/combined_rho*_ops32.csv > "$out/SHA256SUMS"
echo "overnight simulations and aggregation complete"
