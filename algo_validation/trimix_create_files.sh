#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
staining_root="${FLOWMOP_STAINING_ROOT:?Set FLOWMOP_STAINING_ROOT to the staining-trials directory}"
cd "${script_dir}"

## Trimix
python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_a/A05_rep1.fcs" \
        "${staining_root}/stain_a/A3_rep1.fcs" \
        "${staining_root}/stain_a/A1_rep1.fcs" \
    --specific-proportions 0.2 0.2 0.6 \
    --output-file "${staining_root}/A0531_202060_trimix.fcs" \
    --enable-mixing &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_b/B3_rep1.fcs" \
        "${staining_root}/stain_b/B05_rep1.fcs" \
        "${staining_root}/stain_b/B1_rep1.fcs" \
    --specific-proportions 0.40 0.40 0.20 \
    --output-file "${staining_root}/B3051_404020_trimix.fcs" \
    --enable-mixing &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_c/C05_rep1.fcs" \
        "${staining_root}/stain_c/C3_rep1.fcs" \
        "${staining_root}/stain_c/C1_rep1.fcs" \
    --specific-proportions 0.33 0.33 0.33 \
    --output-file "${staining_root}/C0531_333333_trimix.fcs" \
    --enable-mixing &

# Trimix - 3

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_c/C05_rep2.fcs" \
        "${staining_root}/stain_c/C3_rep2.fcs" \
        "${staining_root}/stain_c/C1_rep2.fcs" \
    --specific-proportions 0.33 0.33 0.33 \
    --output-file "${staining_root}/C0531_333333_trimix.fcs" \
    --enable-mixing &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_b/B05_rep2.fcs" \
        "${staining_root}/stain_b/B1_rep2.fcs" \
        "${staining_root}/stain_b/B3_rep2.fcs" \
    --specific-proportions 0.2 0.2 0.6 \
    --output-file "${staining_root}/B0513_202060_trimix.fcs" \
    --enable-mixing &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_a/A05_rep2.fcs" \
        "${staining_root}/stain_a/A1_rep2.fcs" \
        "${staining_root}/stain_a/A3_rep2.fcs" \
    --specific-proportions 0.4 0.4 0.2 \
    --output-file "${staining_root}/A0513_404020_trimix.fcs" \
    --enable-mixing &

wait 
