#!/usr/bin/env bash
set -euo pipefail

BIN="${TEST_SRCDIR}/${TEST_WORKSPACE}/app/ml-algos"
DATA="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/v3"
TMP="${TEST_TMPDIR}"

if [[ ! -x "${BIN}" ]]; then
  echo "binary not found: ${BIN}" >&2
  exit 1
fi

expect_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    echo "expected output to contain '${needle}'" >&2
    echo "${haystack}" >&2
    exit 1
  fi
}

list_output="$("${BIN}" list --task regression)"
expect_contains "${list_output}" "linear"

eval_json="$("${BIN}" eval --task regression --algorithm ridge --data "${DATA}/regression.csv" --target target --json)"
expect_contains "${eval_json}" "\"regression\""

model_path="${TMP}/offset.bundle"
fit_output="$("${BIN}" fit --task classification --algorithm logistic --data "${DATA}/classification_offset.csv" --target label --transformers standard_scaler --model "${model_path}")"
expect_contains "${fit_output}" "Model saved"

inspect_json="$("${BIN}" inspect --model "${model_path}" --json)"
expect_contains "${inspect_json}" "\"class_labels\":[2,5]"

predict_json="$("${BIN}" predict --model "${model_path}" --input "${DATA}/prediction_features.csv" --json)"
expect_contains "${predict_json}" "\"predictions\""

tune_json="$("${BIN}" tune --task classification --algorithm logistic --data "${DATA}/classification_binary.csv" --target label --transformers standard_scaler --cv-folds 4 --json)"
expect_contains "${tune_json}" "\"best_spec\""
