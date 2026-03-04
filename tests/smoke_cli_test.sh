#!/usr/bin/env bash
set -euo pipefail

bin="${1}"
repo_root="${2}"
sample_data="${repo_root}/data/sample_data.csv"
sample_binary="${repo_root}/data/sample_binary_data.csv"
invalid_nonnumeric="${repo_root}/data/sample_invalid_nonnumeric.csv"

expect_contains() {
  local output="$1"
  local pattern="$2"
  local name="$3"
  if [[ "${output}" != *"${pattern}"* ]]; then
    echo "${name}: expected output to contain '${pattern}'"
    echo "actual output:"
    echo "${output}"
    exit 1
  fi
}

help_output="$(${bin} --help)"
expect_contains "${help_output}" "train --task" "help"
expect_contains "${help_output}" "model-info" "help"

list_reg_output="$(${bin} list --task regression)"
expect_contains "${list_reg_output}" "linear" "list-regression"
expect_contains "${list_reg_output}" "knn-regressor" "list-regression"

list_cls_output="$(${bin} list --task classification)"
expect_contains "${list_cls_output}" "logistic" "list-classification"
expect_contains "${list_cls_output}" "softmax" "list-classification"

eval_reg_output="$(${bin} evaluate --task regression --algorithm ridge --data "${sample_data}")"
expect_contains "${eval_reg_output}" "Metric (r2)" "evaluate-regression"

eval_cls_output="$(${bin} evaluate --task classification --algorithm logistic --data "${sample_binary}")"
expect_contains "${eval_cls_output}" "Metric (accuracy)" "evaluate-classification"

tune_output="$(${bin} tune --task regression --algorithm ridge --data "${sample_data}")"
expect_contains "${tune_output}" "Best r2" "tune-ridge"

json_eval="$(${bin} evaluate --task classification --algorithm knn-classifier --data "${sample_binary}" --json)"
expect_contains "${json_eval}" "\"task\":\"classification\"" "json-evaluate"
expect_contains "${json_eval}" "\"metric_name\":\"accuracy\"" "json-evaluate"

if ${bin} evaluate --task regression --algorithm does-not-exist --data "${sample_data}" >/dev/null 2>&1; then
  echo "unknown-algorithm: expected failure"
  exit 1
fi

if ${bin} evaluate --task classification --algorithm logistic --data "${sample_data}" >/dev/null 2>&1; then
  echo "bad-target-shape: expected failure"
  exit 1
fi

if ${bin} evaluate --task regression --algorithm linear --data "${invalid_nonnumeric}" >/dev/null 2>&1; then
  echo "invalid-csv: expected failure"
  exit 1
fi
