#!/usr/bin/env bash
set -euo pipefail

bin="${TEST_SRCDIR}/${TEST_WORKSPACE}/ml-algos"
sample_data="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/sample_data.csv"
sample_binary="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/sample_binary_data.csv"
sample_binary_offset="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/sample_binary_label_offset.csv"
sample_binary_imbalanced="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/sample_binary_imbalanced.csv"
invalid_inconsistent="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/sample_invalid_inconsistent.csv"
invalid_nonnumeric="${TEST_SRCDIR}/${TEST_WORKSPACE}/data/sample_invalid_nonnumeric.csv"

expect_contains() {
  local output="$1"
  local pattern="$2"
  local name="$3"
  if [[ "${output}" != *"${pattern}"* ]]; then
    echo "${name}: expected output to contain '${pattern}'"
    exit 1
  fi
}

expect_not_contains() {
  local output="$1"
  local pattern="$2"
  local name="$3"
  if [[ "${output}" == *"${pattern}"* ]]; then
    echo "${name}: unexpected output containing '${pattern}'"
    exit 1
  fi
}

help_output="$(${bin} --help)"
expect_contains "${help_output}" "Regression algorithms:" "help"

run_output="$(${bin} "${sample_data}")"
expect_contains "${run_output}" "Algorithm" "default"
expect_not_contains "${run_output}" "Accuracy" "default"

cv_output="$(${bin} "${sample_data}" cv)"
expect_contains "${cv_output}" "Skipping classification CV" "cv"

importance_output="$(${bin} "${sample_data}" importance)"
expect_contains "${importance_output}" "Permutation Feature Importance" "importance"

cluster_output="$(${bin} "${sample_data}" cluster)"
expect_contains "${cluster_output}" "Clustering with k=" "cluster"

classifier_output="$(${bin} "${sample_binary}" logistic)"
expect_contains "${classifier_output}" "Accuracy" "classifier"
expect_contains "${classifier_output}" "Macro-F1" "classifier"
expect_contains "${classifier_output}" "Micro-F1" "classifier"
expect_contains "${classifier_output}" "Confusion Matrix" "classifier"
expect_contains "${classifier_output}" "Decision threshold" "classifier"

classifier_offset_output="$(${bin} "${sample_binary_offset}" logistic)"
expect_contains "${classifier_offset_output}" "Accuracy" "classifier-offset"
expect_not_contains "${classifier_offset_output}" "0.0000" "classifier-offset"

classifier_imbalanced_output="$(${bin} "${sample_binary_imbalanced}" logistic)"
expect_contains "${classifier_imbalanced_output}" "Confusion Matrix" "classifier-imbalanced"
expect_contains "${classifier_imbalanced_output}" "  1:" "classifier-imbalanced"

modern_mlp_output="$(${bin} "${sample_binary}" modern-mlp-cls)"
expect_contains "${modern_mlp_output}" "Decision threshold" "modern-mlp-cls"

grid_reg_output="$(${bin} "${sample_data}" gridsearch)"
expect_contains "${grid_reg_output}" "requires binary target" "gridsearch-regression"

grid_cls_output="$(${bin} "${sample_binary}" gridsearch)"
expect_contains "${grid_cls_output}" "Grid Search: Logistic Regression" "gridsearch-binary"

help_sub_output="$(${bin} "${sample_data}" help)"
expect_contains "${help_sub_output}" "Modes:" "help-subcommand"

if ${bin} "${invalid_inconsistent}" >/dev/null 2>&1; then
  echo "invalid_inconsistent: expected non-zero exit"
  exit 1
fi

if ${bin} "${invalid_nonnumeric}" >/dev/null 2>&1; then
  echo "invalid_nonnumeric: expected non-zero exit"
  exit 1
fi

if ${bin} "${TEST_SRCDIR}/${TEST_WORKSPACE}/data/does_not_exist.csv" >/dev/null 2>&1; then
  echo "missing_csv: expected non-zero exit"
  exit 1
fi
