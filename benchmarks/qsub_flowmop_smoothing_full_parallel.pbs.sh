#!/bin/bash

#PBS -N flowmop_fig2_smoothing_full
#PBS -P eu59
#PBS -q expresssr
#PBS -l walltime=01:00:00
#PBS -l ncpus=48
#PBS -l mem=190GB
#PBS -l jobfs=190GB
#PBS -l storage=gdata/eu59+scratch/eu59+gdata/dk92
#PBS -j oe
#PBS -m abe
#PBS -l wd

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load NCI-data-analysis
module load parallel

if [[ -n "${PYTHON_PACKAGE_DIR:-}" ]]; then
    export PYTHONPATH="${PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
FLOWMOP_ROOT="${FLOWMOP_ROOT:?FLOWMOP_ROOT is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
RSCRIPT_BIN="${RSCRIPT_BIN:?RSCRIPT_BIN is required}"
MAD_SMOOTHING_A="${MAD_SMOOTHING_A:?MAD_SMOOTHING_A is required}"
MAD_SMOOTHING_B="${MAD_SMOOTHING_B:?MAD_SMOOTHING_B is required}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
SCRATCH_DIR="${PBS_JOBFS}/flowmop_fig2_smoothing"
WORKERS="${PBS_NCPUS:-48}"

mkdir -p "${OUT_DIR}" "${SCRATCH_DIR}"
TOTAL_INPUTS=$(($(wc -l < "${MANIFEST_PATH}") - 1))

run_one() {
    local index="$1"
    echo "[$(date --iso-8601=seconds)] Running ${index}/${TOTAL_INPUTS}"
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_full_timegating_corrected.py" \
        run-index \
        --manifest "${MANIFEST_PATH}" \
        --index "${index}" \
        --out-dir "${OUT_DIR}" \
        --repo-root "${REPO_ROOT}" \
        --flowmop-root "${FLOWMOP_ROOT}" \
        --python-bin "${PYTHON_BIN}" \
        --rscript "${RSCRIPT_BIN}" \
        --scratch-dir "${SCRATCH_DIR}" \
        --timeout 21600 \
        --algorithms flowmop \
        --mad-smoothing "${MAD_SMOOTHING_A}" "${MAD_SMOOTHING_B}"
}

export -f run_one
export TOTAL_INPUTS PYTHON_BIN REPO_ROOT MANIFEST_PATH OUT_DIR FLOWMOP_ROOT RSCRIPT_BIN SCRATCH_DIR
export MAD_SMOOTHING_A MAD_SMOOTHING_B

echo "Job ID: ${PBS_JOBID:-manual}"
echo "Host: $(hostname)"
echo "Inputs: ${TOTAL_INPUTS}"
echo "Workers: ${WORKERS}"
echo "Smoothing: ${MAD_SMOOTHING_A} ${MAD_SMOOTHING_B}"
echo "Started: $(date --iso-8601=seconds)"

seq 1 "${TOTAL_INPUTS}" | parallel \
    --jobs "${WORKERS}" \
    --line-buffer \
    --joblog "${OUT_DIR}/parallel_joblog.tsv" \
    run_one {}

echo "Completed: $(date --iso-8601=seconds)"
