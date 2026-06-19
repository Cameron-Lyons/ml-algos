#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

if command -v clang-format-23 >/dev/null 2>&1; then
  clang_format=(clang-format-23)
elif command -v clang-format-22 >/dev/null 2>&1; then
  clang_format=(clang-format-22)
elif command -v clang-format >/dev/null 2>&1; then
  clang_format=(clang-format)
else
  echo "clang-format not found" >&2
  exit 1
fi

files=()
while IFS= read -r file; do
  files+=("${file}")
done < <(git ls-files '*.cc' '*.h')

if ((${#files[@]} == 0)); then
  echo "No C/C++ files to check." >&2
  exit 1
fi

"${clang_format[@]}" --dry-run --Werror "${files[@]}"
