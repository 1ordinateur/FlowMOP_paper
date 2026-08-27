#!/bin/bash

#PBS -N flowmop_fig2_default
#PBS -q normal
#PBS -l walltime=12:00:00
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l jobfs=4GB
#PBS -l storage=gdata/PROJECT_CODE+scratch/PROJECT_CODE+gdata/dk92
#PBS -j oe
#PBS -m abe
#PBS -l wd

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load NCI-data-analysis

if [[ -n "${PYTHON_PACKAGE_DIR:-}" ]]; then
    export PYTHONPATH="${PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
FLOWMOP_ROOT="${FLOWMOP_ROOT:?FLOWMOP_ROOT is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
RSCRIPT_BIN="${RSCRIPT_BIN:?RSCRIPT_BIN is required}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
SCRATCH_DIR="${PBS_JOBFS}/flowmop_fig2_default"

mkdir -p "${OUT_DIR}" "${SCRATCH_DIR}"
TOTAL_INPUTS=$(($(wc -l < "${MANIFEST_PATH}") - 1))

echo "Job ID: ${PBS_JOBID:-manual}"
echo "Host: $(hostname)"
echo "Inputs: ${TOTAL_INPUTS}"
echo "Smoothing: 0.01 0.05"
echo "Started: $(date --iso-8601=seconds)"

for INDEX in $(seq 1 "${TOTAL_INPUTS}"); do
    echo "[$(date --iso-8601=seconds)] Running ${INDEX}/${TOTAL_INPUTS}"
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_full_timegating_corrected.py" \
        run-index \
        --manifest "${MANIFEST_PATH}" \
        --index "${INDEX}" \
        --out-dir "${OUT_DIR}" \
        --repo-root "${REPO_ROOT}" \
        --flowmop-root "${FLOWMOP_ROOT}" \
        --python-bin "${PYTHON_BIN}" \
        --rscript "${RSCRIPT_BIN}" \
        --scratch-dir "${SCRATCH_DIR}" \
        --timeout 21600 \
        --algorithms flowmop \
        --mad-smoothing 0.01 0.05
done

echo "Completed: $(date --iso-8601=seconds)"
