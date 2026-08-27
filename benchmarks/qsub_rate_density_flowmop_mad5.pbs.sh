#!/bin/bash

#PBS -N flowmop_timewarp_mad5
#PBS -q expresssr
#PBS -l walltime=01:00:00
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l jobfs=400GB
#PBS -l storage=gdata/PROJECT_CODE+gdata/dk92+scratch/PROJECT_CODE
#PBS -j oe
#PBS -m abe
#PBS -l wd

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load NCI-data-analysis
module load parallel

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
PYTHON_PACKAGE_DIR="${PYTHON_PACKAGE_DIR:?PYTHON_PACKAGE_DIR is required}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
WORKERS="${WORKERS:-30}"

export PYTHONPATH="${PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export DASK_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${OUT_ROOT}"
TASK_FILE="${PBS_JOBFS}/flowmop_timewarp_files.txt"
BASE_FILES=(
    B1B3_1585_bimix.fcs
    C3C1_8020_bimix.fcs
    C1C3_1090_bimix.fcs
    A05A3_1585_bimix.fcs
    B3B1_5050_bimix.fcs
    A05A1_3565_bimix.fcs
    A1A3_6040_bimix.fcs
    A1A3_5050_bimix.fcs
    B1B3_5545_bimix.fcs
    C1C05_6040_bimix.fcs
    B1B3B05_504010_trimix.fcs
    A1A05A3_501535_trimix.fcs
    A3A1A05_305020_trimix.fcs
    A3A1A05_502525_trimix.fcs
    C3C1C05_107020_trimix.fcs
    A05A1A3_405010_trimix.fcs
    A1A3A05_106525_trimix.fcs
    B1B05B3_103060_trimix.fcs
    B1B3B05_651025_trimix.fcs
    C1C3C05_353530_trimix.fcs
    B3B1_8020_segment.fcs
    B05B3_1090_segment.fcs
    B3B05_9010_segment.fcs
    C1C3_2575_segment.fcs
    C05C3_1585_segment.fcs
    A05A1_1090_segment.fcs
    A1A3_5545_segment.fcs
    B1B05_7525_segment.fcs
    B1B3_8515_segment.fcs
    C1C05_9010_segment.fcs
)
printf '%s\n' "${BASE_FILES[@]}" > "${TASK_FILE}"

run_one() {
    local base_file="$1"
    local stem="${base_file%.fcs}"
    local out_dir="${OUT_ROOT}/parts/${stem}"
    echo "[$(date --iso-8601=seconds)] Starting ${base_file}"
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_rate_density_mechanism.py" \
        --dataset-dir "${DATASET_DIR}" \
        --base-files "${base_file}" \
        --events 500000 \
        --out-dir "${out_dir}" \
        --timeout 3600 \
        --algorithms flowmop \
        --mad-factor 5 \
        --timewarp-factors 1,20 \
        --random-chunk-size 25000
    echo "[$(date --iso-8601=seconds)] Completed ${base_file}"
}

export -f run_one
export REPO_ROOT DATASET_DIR OUT_ROOT PYTHON_PACKAGE_DIR PYTHONPATH PYTHON_BIN
export DASK_NUM_WORKERS OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

echo "Job ID: ${PBS_JOBID:-manual}"
echo "Host: $(hostname)"
echo "Inputs: $(wc -l < "${TASK_FILE}")"
echo "Workers: ${WORKERS}"
echo "FlowMOP settings: MAD factor 5; smoothing 0.01,0.05"
echo "Started: $(date --iso-8601=seconds)"

parallel \
    --jobs "${WORKERS}" \
    --line-buffer \
    --joblog "${OUT_ROOT}/parallel_joblog.tsv" \
    run_one :::: "${TASK_FILE}"

echo "Completed: $(date --iso-8601=seconds)"
