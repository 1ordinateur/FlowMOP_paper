#!/bin/bash

#PBS -N flowmop_fig2_capped_grid
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

BASE_DIR="${BASE_DIR:?BASE_DIR is required}"
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
FLOWMOP_ROOT="${FLOWMOP_ROOT:?FLOWMOP_ROOT is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
RSCRIPT_BIN="${RSCRIPT_BIN:?RSCRIPT_BIN is required}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
PYTHON_PACKAGE_DIR="${PYTHON_PACKAGE_DIR:-${BASE_DIR}/python_packages}"
MAD_FACTOR="${MAD_FACTOR:-5}"
WORKERS="${PBS_NCPUS:-48}"
SCRATCH_ROOT="${PBS_JOBFS}/flowmop_smoothing_grid"

export PYTHONPATH="${PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export DASK_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${OUT_ROOT}" "${SCRATCH_ROOT}"
TOTAL_INPUTS=$(($(wc -l < "${MANIFEST_PATH}") - 1))

# Includes the original Table S4 grid, the previously run full-dataset settings,
# and intermediate pairs around the expected sensitivity-specificity knee.
SETTINGS=(
    "0_0 0 0"
    "001_002 0.01 0.02"
    "001_005 0.01 0.05"
    "001_009 0.01 0.09"
    "001_02 0.01 0.20"
    "002_005 0.02 0.05"
    "002_009 0.02 0.09"
    "002_02 0.02 0.20"
    "002_034 0.02 0.34"
    "005_009 0.05 0.09"
    "005_02 0.05 0.20"
    "005_034 0.05 0.34"
    "01_02 0.10 0.20"
    "01_034 0.10 0.34"
    "01_05 0.10 0.50"
    "01_09 0.10 0.90"
    "01_10 0.10 1.00"
    "02_09 0.20 0.90"
    "04_09 0.40 0.90"
)

run_one() {
    local slug="$1"
    local short="$2"
    local long="$3"
    local index="$4"
    local out_dir="${OUT_ROOT}/${slug}"
    local scratch_dir="${SCRATCH_ROOT}/${slug}"

    mkdir -p "${out_dir}" "${scratch_dir}"
    echo "[$(date --iso-8601=seconds)] ${slug} (${short},${long}) input ${index}/${TOTAL_INPUTS}"
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_full_timegating_corrected.py" \
        run-index \
        --manifest "${MANIFEST_PATH}" \
        --index "${index}" \
        --out-dir "${out_dir}" \
        --repo-root "${REPO_ROOT}" \
        --flowmop-root "${FLOWMOP_ROOT}" \
        --python-bin "${PYTHON_BIN}" \
        --rscript "${RSCRIPT_BIN}" \
        --scratch-dir "${scratch_dir}" \
        --timeout 21600 \
        --algorithms flowmop \
        --mad-factor "${MAD_FACTOR}" \
        --mad-smoothing "${short}" "${long}"
}

export -f run_one
export TOTAL_INPUTS PYTHON_BIN REPO_ROOT FLOWMOP_ROOT MANIFEST_PATH OUT_ROOT
export RSCRIPT_BIN SCRATCH_ROOT MAD_FACTOR

TASK_FILE="${PBS_JOBFS}/flowmop_smoothing_grid_tasks.tsv"
: > "${TASK_FILE}"
for setting in "${SETTINGS[@]}"; do
    read -r slug short long <<< "${setting}"
    for index in $(seq 1 "${TOTAL_INPUTS}"); do
        printf '%s\t%s\t%s\t%s\n' "${slug}" "${short}" "${long}" "${index}" >> "${TASK_FILE}"
    done
done

echo "Job ID: ${PBS_JOBID:-manual}"
echo "Inputs: ${TOTAL_INPUTS}"
echo "Settings: ${#SETTINGS[@]}"
echo "Tasks: $(wc -l < "${TASK_FILE}")"
echo "Workers: ${WORKERS}"
echo "MAD factor: ${MAD_FACTOR}"
echo "Started: $(date --iso-8601=seconds)"

parallel \
    --jobs "${WORKERS}" \
    --colsep '\t' \
    --line-buffer \
    --joblog "${OUT_ROOT}/parallel_joblog.tsv" \
    run_one {1} {2} {3} {4} :::: "${TASK_FILE}"

"${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/analyse_flowmop_smoothing_grid.py" \
    --manifest "${MANIFEST_PATH}" \
    --results-root "${OUT_ROOT}" \
    --output-dir "${OUT_ROOT}/analysis" \
    --mad-factor "${MAD_FACTOR}"

echo "Completed: $(date --iso-8601=seconds)"
