#!/usr/bin/env bash
set -euo pipefail

bin="${1}"
repo_root="${2}"
sample_data="${repo_root}/data/sample_data.csv"
sample_binary="${repo_root}/data/sample_binary_data.csv"
sample_binary_offset="${repo_root}/data/sample_binary_label_offset.csv"

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

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

reg_model="${tmpdir}/ridge.model"
cls_model="${tmpdir}/logistic.model"
cls_offset_model="${tmpdir}/logistic_offset.model"
reg_features="${tmpdir}/reg_features.csv"
cls_features="${tmpdir}/cls_features.csv"
cls_offset_features="${tmpdir}/cls_offset_features.csv"

awk -F',' 'BEGIN{OFS=","} {for(i=1;i<NF;i++){printf "%s", $i; if(i<NF-1) printf OFS;} printf ORS;}' "${sample_data}" >"${reg_features}"
awk -F',' 'BEGIN{OFS=","} {for(i=1;i<NF;i++){printf "%s", $i; if(i<NF-1) printf OFS;} printf ORS;}' "${sample_binary}" >"${cls_features}"
awk -F',' 'BEGIN{OFS=","} {for(i=1;i<NF;i++){printf "%s", $i; if(i<NF-1) printf OFS;} printf ORS;}' "${sample_binary_offset}" >"${cls_offset_features}"

train_reg_output="$(${bin} train --task regression --algorithm ridge --data "${sample_data}" --model "${reg_model}")"
expect_contains "${train_reg_output}" "Model saved" "train-regression"

train_cls_output="$(${bin} train --task classification --algorithm logistic --data "${sample_binary}" --model "${cls_model}")"
expect_contains "${train_cls_output}" "Model saved" "train-classification"

train_cls_offset_output="$(${bin} train --task classification --algorithm logistic --data "${sample_binary_offset}" --model "${cls_offset_model}")"
expect_contains "${train_cls_offset_output}" "Model saved" "train-classification-offset"

info_reg="$(${bin} model-info --model "${reg_model}")"
expect_contains "${info_reg}" "Algorithm: ridge" "info-regression"
expect_contains "${info_reg}" "Task: regression" "info-regression"

info_cls_json="$(${bin} model-info --model "${cls_model}" --json)"
expect_contains "${info_cls_json}" "\"algorithm\":\"logistic\"" "info-classification-json"
expect_contains "${info_cls_json}" "\"task\":\"classification\"" "info-classification-json"

predict_reg_output="$(${bin} predict --model "${reg_model}" --input "${reg_features}")"
expect_contains "${predict_reg_output}" "Predictions:" "predict-regression"

predict_cls_json="$(${bin} predict --model "${cls_model}" --input "${cls_features}" --json)"
expect_contains "${predict_cls_json}" "\"predictions\":" "predict-classification-json"

predict_cls_offset_json="$(${bin} predict --model "${cls_offset_model}" --input "${cls_offset_features}" --json)"
expect_contains "${predict_cls_offset_json}" "\"predictions\":[10" "predict-classification-offset-json"

if ${bin} predict --model "${reg_model}" --input "${cls_features}" >/dev/null 2>&1; then
  echo "predict-width-mismatch: expected failure"
  exit 1
fi
