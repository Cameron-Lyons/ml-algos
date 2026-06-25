#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

if command -v clang-tidy-23 >/dev/null 2>&1; then
  clang_tidy=(clang-tidy-23)
elif command -v clang-tidy-22 >/dev/null 2>&1; then
  clang_tidy=(clang-tidy-22)
elif command -v clang-tidy >/dev/null 2>&1; then
  clang_tidy=(clang-tidy)
else
  echo "clang-tidy not found" >&2
  exit 1
fi

extra_args=(
  -std=c++26
  -stdlib=libc++
  -I.
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wsign-conversion
)

status=0
while IFS= read -r file; do
  if ! "${clang_tidy[@]}" --quiet "${file}" --config-file=.clang-tidy -- "${extra_args[@]}"; then
    status=1
  fi
done < <(git ls-files '*.cc')

exit "${status}"
