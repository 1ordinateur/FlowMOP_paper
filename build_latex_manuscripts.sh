#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pandoc_bin="${PANDOC_BIN:-pandoc}"
tectonic_bin="${TECTONIC_BIN:-tectonic}"
cairosvg_bin="${CAIROSVG_BIN:-cairosvg}"

pandoc_path() {
  local path="$1"
  if [[ "${pandoc_bin}" == *.exe ]]; then
    wslpath -w "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

build_vector_figures() {
  MPLCONFIGDIR="${TMPDIR:-/tmp}/flowmop-matplotlib" \
    python3 "${script_dir}/figs_data/harmonize_figures_2_3.py"

  local figure_number
  for figure_number in 2 3 5 6 7; do
    "${cairosvg_bin}" \
      "${script_dir}/figs_data/figure_${figure_number}.svg" \
      --output "${script_dir}/figs_data/figure_${figure_number}.pdf"
  done
  local figure_panel
  for figure_panel in \
    figure_2_panel_a \
    figure_2_panel_b \
    figure_3_panel_a \
    figure_3_panel_b \
    figure_3_panel_cd; do
    "${cairosvg_bin}" \
      "${script_dir}/figs_data/${figure_panel}.svg" \
      --output "${script_dir}/figs_data/${figure_panel}.pdf"
  done
  python3 "${script_dir}/figs_data/fig_4_data/harmonize_figure_4.py"
  "${cairosvg_bin}" \
    "${script_dir}/figs_data/figure_4_harmonized.svg" \
    --output "${script_dir}/figs_data/figure_4_harmonized.pdf"
  "${cairosvg_bin}" \
    "${script_dir}/figs_data/figure_4_panel_a.svg" \
    --output "${script_dir}/figs_data/figure_4_panel_a.pdf"
  "${cairosvg_bin}" \
    "${script_dir}/figs_data/figure_4_panel_bc.svg" \
    --output "${script_dir}/figs_data/figure_4_panel_bc.pdf"
  "${cairosvg_bin}" \
    "${script_dir}/figs_data/Supp_fig_8.svg" \
    --output "${script_dir}/figs_data/Supp_fig_8.pdf"
}

build_manuscript() {
  local input_file="$1"
  local output_stem="$2"
  local subtitle="$3"
  local tracked="$4"

  local metadata_args=(
    --metadata "title=FlowMOP: An Automated Flow Cytometry Time, Debris, and Doublet Removal Tool"
    --metadata "subtitle=${subtitle}"
  )

  if [[ "${tracked}" == "true" ]]; then
    metadata_args+=(--metadata tracked=true)
  fi

  local pandoc_root
  local pandoc_template
  local pandoc_filter
  local pandoc_input
  local pandoc_output
  pandoc_root="$(pandoc_path "${script_dir}")"
  pandoc_template="$(pandoc_path "${script_dir}/latex/flowmop-template.tex")"
  pandoc_filter="$(pandoc_path "${script_dir}/latex/flowmop-format.lua")"
  pandoc_input="$(pandoc_path "${script_dir}/${input_file}")"
  pandoc_output="$(pandoc_path "${script_dir}/${output_stem}.tex")"

  "${pandoc_bin}" \
    --from=gfm-implicit_figures \
    --to=latex \
    --standalone \
    --number-sections \
    --resource-path="${pandoc_root}" \
    --template="${pandoc_template}" \
    --lua-filter="${pandoc_filter}" \
    "${metadata_args[@]}" \
    "${pandoc_input}" \
    --output="${pandoc_output}"

  "${tectonic_bin}" \
    --chatter minimal \
    --outdir "${script_dir}" \
    "${script_dir}/${output_stem}.tex"
}

build_vector_figures

build_manuscript \
  "FlowMOP_submission.md" \
  "FlowMOP_submission" \
  "Revised manuscript" \
  "false"

build_manuscript \
  "FlowMOP_submission_tracked.md" \
  "FlowMOP_submission_tracked" \
  "Revised manuscript with tracked changes" \
  "true"
