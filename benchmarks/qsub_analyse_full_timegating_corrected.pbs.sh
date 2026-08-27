#!/bin/bash

#PBS -N flowmop_fig2_analyse
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l jobfs=1GB
#PBS -l storage=gdata/PROJECT_CODE+gdata/dk92
#PBS -j oe
#PBS -m abe
#PBS -l wd

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load NCI-data-analysis

if [[ -n "${PYTHON_PACKAGE_DIR:-}" ]]; then
    export PYTHONPATH="${PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export MPLCONFIGDIR="${PBS_JOBFS}/matplotlib"

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
ANALYSIS_DIR="${ANALYSIS_DIR:?ANALYSIS_DIR is required}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

mkdir -p "${MPLCONFIGDIR}" "${ANALYSIS_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_full_timegating_corrected.py" \
    analyse \
    --manifest "${MANIFEST_PATH}" \
    --results-dir "${RESULTS_DIR}" \
    --out-dir "${ANALYSIS_DIR}"
