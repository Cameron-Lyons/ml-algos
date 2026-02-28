#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tests/benchmark_runtime.sh [options]

Options:
  --baseline-repo <path>   Optional repo path for "before" measurements.
  --candidate-repo <path>  Repo path for "after" measurements (default: current repo).
  --runs <n>               Timed runs per benchmark (default: 5).
  --warmup <n>             Warmup runs per benchmark (default: 1).
  --output-dir <path>      Write CSV+JSON outputs in this directory.
  --csv-out <path>         Write benchmark results as CSV.
  --json-out <path>        Write benchmark results as JSON.
  --skip-build             Skip CMake build step and use existing binaries.
  -h, --help               Show this help message.

Examples:
  tests/benchmark_runtime.sh --runs 8 --warmup 2
  tests/benchmark_runtime.sh \
    --baseline-repo /path/to/ml-algos-before \
    --candidate-repo /path/to/ml-algos-after \
    --runs 8 --warmup 2
  tests/benchmark_runtime.sh --output-dir /tmp/ml-bench --runs 8 --warmup 2
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate_repo="${repo_root}"
baseline_repo=""
runs=5
warmup=1
build_enabled=1
output_dir=""
csv_out=""
json_out=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --baseline-repo)
    baseline_repo="$2"
    shift 2
    ;;
  --candidate-repo)
    candidate_repo="$2"
    shift 2
    ;;
  --runs)
    runs="$2"
    shift 2
    ;;
  --warmup)
    warmup="$2"
    shift 2
    ;;
  --output-dir)
    output_dir="$2"
    shift 2
    ;;
  --csv-out)
    csv_out="$2"
    shift 2
    ;;
  --json-out)
    json_out="$2"
    shift 2
    ;;
  --skip-build)
    build_enabled=0
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if [[ -n "$output_dir" ]]; then
  mkdir -p "$output_dir"
  if [[ -z "$csv_out" ]]; then
    csv_out="$output_dir/benchmark_runtime.csv"
  fi
  if [[ -z "$json_out" ]]; then
    json_out="$output_dir/benchmark_runtime.json"
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for timing." >&2
  exit 1
fi
if ! command -v c++ >/dev/null 2>&1; then
  echo "c++ is required for the tSNE benchmark harness." >&2
  exit 1
fi

measure_command_ms() {
  local runs_local="$1"
  local warmup_local="$2"
  shift 2
  python3 - "$runs_local" "$warmup_local" "$@" <<'PY'
import statistics
import subprocess
import sys
import time

runs = int(sys.argv[1])
warmup = int(sys.argv[2])
cmd = sys.argv[3:]

devnull = subprocess.DEVNULL
for _ in range(warmup):
    subprocess.run(cmd, stdout=devnull, stderr=devnull, check=True)

samples = []
for _ in range(runs):
    t0 = time.perf_counter()
    subprocess.run(cmd, stdout=devnull, stderr=devnull, check=True)
    samples.append((time.perf_counter() - t0) * 1000.0)

mean = statistics.fmean(samples)
median = statistics.median(samples)
minimum = min(samples)
maximum = max(samples)
stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
print(f"{mean:.3f}\t{median:.3f}\t{minimum:.3f}\t{maximum:.3f}\t{stdev:.3f}")
PY
}

build_binary() {
  local repo="$1"
  if [[ ! -d "$repo" ]]; then
    echo "Repo path does not exist: $repo" >&2
    exit 1
  fi
  if [[ "$build_enabled" -eq 1 ]]; then
    cmake -S "$repo" -B "$repo/build" >/dev/null
    cmake --build "$repo/build" >/dev/null
  fi
  local bin="$repo/build/ml-algos"
  if [[ ! -x "$bin" ]]; then
    echo "Missing executable: $bin" >&2
    echo "Build first or omit --skip-build." >&2
    exit 1
  fi
  printf "%s" "$bin"
}

build_tsne_harness() {
  local repo="$1"
  local out_dir="$2"
  local src_file="$out_dir/tsne_bench.cpp"
  local exe_file="$out_dir/tsne_bench"

  cat >"$src_file" <<'CPP'
#include <iostream>
#include <random>

#include "src/matrix.cpp"
#include "src/unsupervised/tsne.cpp"

int main() {
  constexpr size_t n = 240;
  constexpr size_t d = 12;
  constexpr size_t out_dims = 2;
  constexpr int iterations = 250;
  constexpr double learning_rate = 100.0;
  constexpr double sigma = 1.0;

  Points X(n, Point(d, 0.0));
  std::mt19937 rng(123);
  std::normal_distribution<double> c1(0.0, 1.0);
  std::normal_distribution<double> c2(5.0, 1.2);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      X[i][j] = (i < n / 2) ? c1(rng) : c2(rng);
    }
  }

  Points Y = tSNE(X, out_dims, iterations, learning_rate, sigma);

  // Keep output small and deterministic so the compiler cannot elide work.
  double checksum = 0.0;
  for (const auto &p : Y) {
    checksum += p[0] * 0.5 + p[1] * 0.25;
  }
  std::cout << checksum << "\n";
  return 0;
}
CPP

  c++ -std=c++23 -O3 -I"$repo" "$src_file" -o "$exe_file"
  printf "%s" "$exe_file"
}

stats_field() {
  local stats="$1"
  local field="$2"
  printf "%s" "$stats" | awk -F'\t' -v idx="$field" '{print $idx}'
}

compute_delta_pct_number() {
  local base="$1"
  local cand="$2"
  awk -v b="$base" -v c="$cand" 'BEGIN {
    if (b == 0) { print ""; exit 0; }
    printf "%.6f", ((c - b) / b) * 100.0;
  }'
}

compute_speedup_number() {
  local base="$1"
  local cand="$2"
  awk -v b="$base" -v c="$cand" 'BEGIN {
    if (c == 0) { print ""; exit 0; }
    printf "%.6f", b / c;
  }'
}

format_pct() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo "n/a"
    return
  fi
  awk -v v="$value" 'BEGIN { printf "%.2f%%", v; }'
}

format_speedup() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo "n/a"
    return
  fi
  awk -v v="$value" 'BEGIN { printf "%.3fx", v; }'
}

write_machine_outputs() {
  if [[ -z "$csv_out" && -z "$json_out" ]]; then
    return
  fi

  if [[ -n "$csv_out" ]]; then
    mkdir -p "$(dirname "$csv_out")"
  fi
  if [[ -n "$json_out" ]]; then
    mkdir -p "$(dirname "$json_out")"
  fi

  BENCH_TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  export BENCH_TIMESTAMP_UTC
  export BENCH_RUNS="$runs"
  export BENCH_WARMUP="$warmup"
  export BENCH_BUILD_ENABLED="$build_enabled"
  export BENCH_BASELINE_REPO="$baseline_repo"
  export BENCH_CANDIDATE_REPO="$candidate_repo"

  export BASE_GRID_MEAN="$baseline_grid_mean"
  export BASE_GRID_MEDIAN="$baseline_grid_median"
  export BASE_GRID_MIN="$baseline_grid_min"
  export BASE_GRID_MAX="$baseline_grid_max"
  export BASE_GRID_STDEV="$baseline_grid_stdev"

  export BASE_TSNE_MEAN="$baseline_tsne_mean"
  export BASE_TSNE_MEDIAN="$baseline_tsne_median"
  export BASE_TSNE_MIN="$baseline_tsne_min"
  export BASE_TSNE_MAX="$baseline_tsne_max"
  export BASE_TSNE_STDEV="$baseline_tsne_stdev"

  export CAND_GRID_MEAN="$candidate_grid_mean"
  export CAND_GRID_MEDIAN="$candidate_grid_median"
  export CAND_GRID_MIN="$candidate_grid_min"
  export CAND_GRID_MAX="$candidate_grid_max"
  export CAND_GRID_STDEV="$candidate_grid_stdev"

  export CAND_TSNE_MEAN="$candidate_tsne_mean"
  export CAND_TSNE_MEDIAN="$candidate_tsne_median"
  export CAND_TSNE_MIN="$candidate_tsne_min"
  export CAND_TSNE_MAX="$candidate_tsne_max"
  export CAND_TSNE_STDEV="$candidate_tsne_stdev"

  export GRID_DELTA_NUM="$grid_delta_num"
  export TSNE_DELTA_NUM="$tsne_delta_num"
  export GRID_SPEEDUP_NUM="$grid_speedup_num"
  export TSNE_SPEEDUP_NUM="$tsne_speedup_num"

  python3 - "$csv_out" "$json_out" <<'PY'
import csv
import json
import os
import sys

csv_out = sys.argv[1]
json_out = sys.argv[2]

def env(name, default=""):
    return os.environ.get(name, default)

def to_float(value):
    if value in ("", None):
        return None
    return float(value)

def stats(prefix):
    return {
        "mean_ms": to_float(env(f"{prefix}_MEAN")),
        "median_ms": to_float(env(f"{prefix}_MEDIAN")),
        "min_ms": to_float(env(f"{prefix}_MIN")),
        "max_ms": to_float(env(f"{prefix}_MAX")),
        "stdev_ms": to_float(env(f"{prefix}_STDEV")),
    }

rows = []
baseline_repo = env("BENCH_BASELINE_REPO")
candidate_repo = env("BENCH_CANDIDATE_REPO")
common = {
    "timestamp_utc": env("BENCH_TIMESTAMP_UTC"),
    "runs": int(env("BENCH_RUNS")),
    "warmup": int(env("BENCH_WARMUP")),
    "build_enabled": int(env("BENCH_BUILD_ENABLED")),
    "baseline_repo": baseline_repo,
    "candidate_repo": candidate_repo,
}

if baseline_repo:
    for benchmark, prefix in (
        ("gridsearch", "BASE_GRID"),
        ("tsne-harness", "BASE_TSNE"),
    ):
        s = stats(prefix)
        rows.append(
            {
                **common,
                "benchmark": benchmark,
                "label": "baseline",
                **s,
                "delta_pct": None,
                "speedup": None,
            }
        )

for benchmark, prefix, dkey, skey in (
    ("gridsearch", "CAND_GRID", "GRID_DELTA_NUM", "GRID_SPEEDUP_NUM"),
    ("tsne-harness", "CAND_TSNE", "TSNE_DELTA_NUM", "TSNE_SPEEDUP_NUM"),
):
    s = stats(prefix)
    rows.append(
        {
            **common,
            "benchmark": benchmark,
            "label": "candidate",
            **s,
            "delta_pct": to_float(env(dkey)),
            "speedup": to_float(env(skey)),
        }
    )

if csv_out:
    fieldnames = [
        "timestamp_utc",
        "benchmark",
        "label",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "stdev_ms",
        "runs",
        "warmup",
        "build_enabled",
        "baseline_repo",
        "candidate_repo",
        "delta_pct",
        "speedup",
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

json_obj = {
    "config": common,
    "candidate": {
        "gridsearch": stats("CAND_GRID"),
        "tsne_harness": stats("CAND_TSNE"),
    },
}
if baseline_repo:
    json_obj["baseline"] = {
        "gridsearch": stats("BASE_GRID"),
        "tsne_harness": stats("BASE_TSNE"),
    }
    json_obj["comparison"] = {
        "gridsearch": {
            "delta_pct": to_float(env("GRID_DELTA_NUM")),
            "speedup": to_float(env("GRID_SPEEDUP_NUM")),
        },
        "tsne_harness": {
            "delta_pct": to_float(env("TSNE_DELTA_NUM")),
            "speedup": to_float(env("TSNE_SPEEDUP_NUM")),
        },
    }

if json_out:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)
PY

  if [[ -n "$csv_out" ]]; then
    echo "Wrote CSV:  $csv_out"
  fi
  if [[ -n "$json_out" ]]; then
    echo "Wrote JSON: $json_out"
  fi
}

write_summary_and_outputs() {
  if [[ -n "$baseline_repo" ]]; then
    grid_delta_num="$(compute_delta_pct_number "$baseline_grid_mean" "$candidate_grid_mean")"
    tsne_delta_num="$(compute_delta_pct_number "$baseline_tsne_mean" "$candidate_tsne_mean")"
    grid_speedup_num="$(compute_speedup_number "$baseline_grid_mean" "$candidate_grid_mean")"
    tsne_speedup_num="$(compute_speedup_number "$baseline_tsne_mean" "$candidate_tsne_mean")"
    grid_delta="$(format_pct "$grid_delta_num")"
    tsne_delta="$(format_pct "$tsne_delta_num")"
    grid_speedup="$(format_speedup "$grid_speedup_num")"
    tsne_speedup="$(format_speedup "$tsne_speedup_num")"

    echo
    echo "Summary (mean runtime in ms; negative delta is faster):"
    printf "%-18s %-14s %-14s %-12s %-10s\n" "Benchmark" "Baseline" "Candidate" "Delta" "Speedup"
    printf "%-18s %-14s %-14s %-12s %-10s\n" "gridsearch" "$baseline_grid_mean" "$candidate_grid_mean" "$grid_delta" "$grid_speedup"
    printf "%-18s %-14s %-14s %-12s %-10s\n" "tsne-harness" "$baseline_tsne_mean" "$candidate_tsne_mean" "$tsne_delta" "$tsne_speedup"
  else
    grid_delta_num=""
    tsne_delta_num=""
    grid_speedup_num=""
    tsne_speedup_num=""

    echo
    echo "Single-repo summary (mean runtime in ms):"
    printf "%-18s %-14s\n" "Benchmark" "Candidate"
    printf "%-18s %-14s\n" "gridsearch" "$candidate_grid_mean"
    printf "%-18s %-14s\n" "tsne-harness" "$candidate_tsne_mean"
  fi

  write_machine_outputs
}

declare -a tmp_dirs=()
cleanup() {
  if [[ ${#tmp_dirs[@]} -eq 0 ]]; then
    return
  fi
  for dir in "${tmp_dirs[@]}"; do
    rm -rf "$dir" 2>/dev/null || true
  done
}
trap cleanup EXIT

run_suite() {
  local repo="$1"
  local label="$2"

  local data_file="$repo/data/sample_data.csv"
  if [[ ! -f "$data_file" ]]; then
    echo "Missing dataset: $data_file" >&2
    exit 1
  fi

  local bin
  bin="$(build_binary "$repo")"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  tmp_dirs+=("$tmp_dir")
  local tsne_exe
  tsne_exe="$(build_tsne_harness "$repo" "$tmp_dir")"

  local grid_stats
  grid_stats="$(measure_command_ms "$runs" "$warmup" "$bin" "$data_file" "gridsearch")"
  local tsne_stats
  tsne_stats="$(measure_command_ms "$runs" "$warmup" "$tsne_exe")"

  echo "[$label] gridsearch(ms): $grid_stats"
  echo "[$label] tsne(ms):      $tsne_stats"
}

echo "Benchmark config: runs=$runs warmup=$warmup build=$build_enabled"

candidate_grid_stats=""
candidate_tsne_stats=""
baseline_grid_stats=""
baseline_tsne_stats=""

baseline_grid_mean=""
baseline_grid_median=""
baseline_grid_min=""
baseline_grid_max=""
baseline_grid_stdev=""
baseline_tsne_mean=""
baseline_tsne_median=""
baseline_tsne_min=""
baseline_tsne_max=""
baseline_tsne_stdev=""

candidate_grid_mean=""
candidate_grid_median=""
candidate_grid_min=""
candidate_grid_max=""
candidate_grid_stdev=""
candidate_tsne_mean=""
candidate_tsne_median=""
candidate_tsne_min=""
candidate_tsne_max=""
candidate_tsne_stdev=""

grid_delta_num=""
tsne_delta_num=""
grid_speedup_num=""
tsne_speedup_num=""

if [[ -n "$baseline_repo" ]]; then
  baseline_out="$(run_suite "$baseline_repo" "baseline")"
  echo "$baseline_out"
  baseline_grid_stats="$(echo "$baseline_out" | awk -F': ' '/gridsearch/ {print $2}')"
  baseline_tsne_stats="$(echo "$baseline_out" | awk -F': ' '/tsne/ {print $2}')"
  baseline_grid_mean="$(stats_field "$baseline_grid_stats" 1)"
  baseline_grid_median="$(stats_field "$baseline_grid_stats" 2)"
  baseline_grid_min="$(stats_field "$baseline_grid_stats" 3)"
  baseline_grid_max="$(stats_field "$baseline_grid_stats" 4)"
  baseline_grid_stdev="$(stats_field "$baseline_grid_stats" 5)"
  baseline_tsne_mean="$(stats_field "$baseline_tsne_stats" 1)"
  baseline_tsne_median="$(stats_field "$baseline_tsne_stats" 2)"
  baseline_tsne_min="$(stats_field "$baseline_tsne_stats" 3)"
  baseline_tsne_max="$(stats_field "$baseline_tsne_stats" 4)"
  baseline_tsne_stdev="$(stats_field "$baseline_tsne_stats" 5)"
fi

candidate_out="$(run_suite "$candidate_repo" "candidate")"
echo "$candidate_out"
candidate_grid_stats="$(echo "$candidate_out" | awk -F': ' '/gridsearch/ {print $2}')"
candidate_tsne_stats="$(echo "$candidate_out" | awk -F': ' '/tsne/ {print $2}')"
candidate_grid_mean="$(stats_field "$candidate_grid_stats" 1)"
candidate_grid_median="$(stats_field "$candidate_grid_stats" 2)"
candidate_grid_min="$(stats_field "$candidate_grid_stats" 3)"
candidate_grid_max="$(stats_field "$candidate_grid_stats" 4)"
candidate_grid_stdev="$(stats_field "$candidate_grid_stats" 5)"
candidate_tsne_mean="$(stats_field "$candidate_tsne_stats" 1)"
candidate_tsne_median="$(stats_field "$candidate_tsne_stats" 2)"
candidate_tsne_min="$(stats_field "$candidate_tsne_stats" 3)"
candidate_tsne_max="$(stats_field "$candidate_tsne_stats" 4)"
candidate_tsne_stdev="$(stats_field "$candidate_tsne_stats" 5)"

write_summary_and_outputs
