#!/bin/bash

#PBS -N flowmop_fig2_corrected
#PBS -q normal
#PBS -l walltime=06:00:00
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
if [[ -n "${R_PACKAGE_DIR:-}" ]]; then
    export R_LIBS_USER="${R_PACKAGE_DIR}"
fi

REPO_ROOT="${REPO_ROOT:?REPO_ROOT must point to the staged FlowMOP_paper benchmark directory}"
FLOWMOP_ROOT="${FLOWMOP_ROOT:?FLOWMOP_ROOT must point to the staged FlowMOP checkout}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ARRAY_INDEX="${PBS_ARRAY_INDEX:-${ARRAY_INDEX:-}}"

if [[ -z "${ARRAY_INDEX}" ]]; then
    echo "PBS_ARRAY_INDEX or ARRAY_INDEX is required" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
RSCRIPT_BIN="${RSCRIPT_BIN:-$(command -v Rscript)}"
SCRATCH_DIR="${PBS_JOBFS}/flowmop_fig2_corrected"

mkdir -p "${SCRATCH_DIR}" "${OUT_DIR}"

echo "Job ID: ${PBS_JOBID:-manual}"
echo "Array index: ${ARRAY_INDEX}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Python: ${PYTHON_BIN}"
echo "Rscript: ${RSCRIPT_BIN}"

"${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_full_timegating_corrected.py" \
    run-index \
    --manifest "${MANIFEST_PATH}" \
    --index "${ARRAY_INDEX}" \
    --out-dir "${OUT_DIR}" \
    --repo-root "${REPO_ROOT}" \
    --flowmop-root "${FLOWMOP_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --rscript "${RSCRIPT_BIN}" \
    --scratch-dir "${SCRATCH_DIR}" \
    --timeout 21600

echo "Completed: $(date --iso-8601=seconds)"
