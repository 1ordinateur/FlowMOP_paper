#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
staining_root="${FLOWMOP_STAINING_ROOT:?Set FLOWMOP_STAINING_ROOT to the staining-trials directory}"
cd "${script_dir}"

## Segmented 0.5 - 1 
python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_a/A05_rep1.fcs" \
        "${staining_root}/stain_a/A1_rep1.fcs" \
    --specific-proportions 0.2 0.8 \
    --output-file "${staining_root}/A051_2080_segment.fcs" &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_b/B05_rep1.fcs" \
        "${staining_root}/stain_b/B1_rep1.fcs" \
    --specific-proportions 0.65 0.35 \
    --output-file "${staining_root}/B051_6535_segment.fcs" &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_c/C05_rep1.fcs" \
        "${staining_root}/stain_c/C1_rep1.fcs" \
    --specific-proportions 0.50 0.50 \
    --output-file "${staining_root}/C051_5050_segment.fcs" &

# Segmented 0.5 - 3

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_c/C05_rep2.fcs" \
        "${staining_root}/stain_c/C3_rep1.fcs" \
    --specific-proportions 0.50 0.50 \
    --output-file "${staining_root}/C053_5050_segment.fcs" &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_b/B05_rep2.fcs" \
        "${staining_root}/stain_b/B3_rep1.fcs" \
    --specific-proportions 0.2 0.8 \
    --output-file "${staining_root}/B053_2080_segment.fcs" &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_a/A05_rep2.fcs" \
        "${staining_root}/stain_a/A3_rep1.fcs" \
    --specific-proportions 0.65 0.35 \
    --output-file "${staining_root}/A053_6535_segment.fcs" &

# Segmented 1 - 3

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_a/A3_rep2.fcs" \
        "${staining_root}/stain_a/A1_rep2.fcs" \
    --specific-proportions 0.65 0.35 \
    --output-file "${staining_root}/A31_6535_segment.fcs" &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_c/C3_rep2.fcs" \
        "${staining_root}/stain_c/C1_rep2.fcs" \
    --specific-proportions 0.50 0.50 \
    --output-file "${staining_root}/C31_5050_segment.fcs" &

python time_gate_executables/concatenate_fcs_cli.py \
    --specific-files \
        "${staining_root}/stain_b/B3_rep2.fcs" \
        "${staining_root}/stain_b/B1_rep2.fcs" \
    --specific-proportions 0.80 0.20 \
    --output-file "${staining_root}/B31_8020_segment.fcs" &

wait 
