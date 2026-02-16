#!/usr/bin/env bash
set -euo pipefail

bin="${TEST_SRCDIR}/${TEST_WORKSPACE}/ml-algos"
sample_data="${TEST_SRCDIR}/${TEST_WORKSPACE}/sample_data.csv"
invalid_csv="${TEST_SRCDIR}/${TEST_WORKSPACE}/sample_invalid_nonnumeric.csv"

help_output="$(${bin} --help)"
if [[ "${help_output}" != *"Regression algorithms:"* ]]; then
  echo "expected --help to include regression list"
  exit 1
fi

run_output="$(${bin} "${sample_data}")"
if [[ "${run_output}" != *"Algorithm"* ]]; then
  echo "expected default run output table header"
  exit 1
fi

if ${bin} "${invalid_csv}" >/dev/null 2>&1; then
  echo "expected invalid CSV run to fail"
  exit 1
fi
