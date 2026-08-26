#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pandoc_bin="${PANDOC_BIN:-pandoc}"
tectonic_bin="${TECTONIC_BIN:-tectonic}"
input_stem="${1:-reviewer_response_point_by_point}"
output_stem="${2:-${input_stem}}"

pandoc_path() {
  local path="$1"
  if [[ "${pandoc_bin}" == *.exe ]]; then
    wslpath -w "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

pandoc_root="$(pandoc_path "${script_dir}")"
pandoc_header="$(pandoc_path "${script_dir}/latex/reviewer-response-header.tex")"
pandoc_filter="$(pandoc_path "${script_dir}/latex/reviewer-response-format.lua")"
pandoc_input="$(pandoc_path "${script_dir}/${input_stem}.md")"
pandoc_output="$(pandoc_path "${script_dir}/${output_stem}.tex")"

"${pandoc_bin}" \
  --from=gfm \
  --to=latex \
  --standalone \
  --resource-path="${pandoc_root}" \
  --include-in-header="${pandoc_header}" \
  --lua-filter="${pandoc_filter}" \
  --variable papersize=a4 \
  --variable geometry:margin=25mm \
  --variable fontsize=11pt \
  --variable colorlinks=true \
  --variable linkcolor=blue \
  --variable urlcolor=blue \
  "${pandoc_input}" \
  --output="${pandoc_output}"

"${tectonic_bin}" \
  --chatter minimal \
  --outdir "${script_dir}" \
  "${script_dir}/${output_stem}.tex"
