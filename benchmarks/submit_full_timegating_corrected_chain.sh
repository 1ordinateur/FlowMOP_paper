#!/bin/bash

set -euo pipefail

RUN_ROOT="${1:-/g/data/PROJECT_CODE/flowmop_fig2_corrected_20260808}"
REPO_ROOT="${RUN_ROOT}/repo"
FLOWMOP_ROOT="${REPO_ROOT}/FlowMOP"
PYTHON_PACKAGE_DIR="${RUN_ROOT}/python_packages"
RSCRIPT_BIN="${RSCRIPT_BIN_OVERRIDE:-${RUN_ROOT}/r_env/bin/Rscript}"
LOG_DIR="${RUN_ROOT}/pbs_logs"

PREFLIGHT_MANIFEST="${RUN_ROOT}/manifests/preflight.csv"
FULL_MANIFEST="${RUN_ROOT}/manifests/full.csv"
PREFLIGHT_RESULTS="${RUN_ROOT}/results/preflight_v3"
FULL_RESULTS="${RUN_ROOT}/results/full"
ANALYSIS_DIR="${RUN_ROOT}/analysis"

mkdir -p \
    "${LOG_DIR}" \
    "${PREFLIGHT_RESULTS}" \
    "${FULL_RESULTS}" \
    "${ANALYSIS_DIR}"

if [[ ! -x "${RSCRIPT_BIN}" ]]; then
    echo "Rscript is not executable: ${RSCRIPT_BIN}" >&2
    exit 2
fi

if [[ -n "${PREFLIGHT_JOB_OVERRIDE:-}" ]]; then
    R_DEPS_JOB="reused"
    PREFLIGHT_JOB="${PREFLIGHT_JOB_OVERRIDE}"
else
    R_DEPS_JOB="preinstalled"
    PREFLIGHT_JOB=$(qsub \
        -r y \
        -J 1-6%6 \
        -o "${LOG_DIR}" \
        -v "REPO_ROOT=${REPO_ROOT},FLOWMOP_ROOT=${FLOWMOP_ROOT},MANIFEST_PATH=${PREFLIGHT_MANIFEST},OUT_DIR=${PREFLIGHT_RESULTS},PYTHON_PACKAGE_DIR=${PYTHON_PACKAGE_DIR},RSCRIPT_BIN=${RSCRIPT_BIN}" \
        "${REPO_ROOT}/benchmarks/qsub_full_timegating_corrected.pbs.sh")
fi

FULL_JOBS=()
for START_INDEX in $(seq 1 10 179); do
    END_INDEX=$((START_INDEX + 9))
    if ((END_INDEX > 179)); then
        END_INDEX=179
    fi
    FULL_JOBS+=("$(qsub \
        -r y \
        -J "${START_INDEX}-${END_INDEX}%10" \
        -W "depend=afterok:${PREFLIGHT_JOB}" \
        -o "${LOG_DIR}" \
        -v "REPO_ROOT=${REPO_ROOT},FLOWMOP_ROOT=${FLOWMOP_ROOT},MANIFEST_PATH=${FULL_MANIFEST},OUT_DIR=${FULL_RESULTS},PYTHON_PACKAGE_DIR=${PYTHON_PACKAGE_DIR},RSCRIPT_BIN=${RSCRIPT_BIN}" \
        "${REPO_ROOT}/benchmarks/qsub_full_timegating_corrected.pbs.sh")")
done

FULL_DEPENDENCIES=$(IFS=:; echo "${FULL_JOBS[*]}")

ANALYSIS_JOB=$(qsub \
    -r y \
    -W "depend=afterok:${FULL_DEPENDENCIES}" \
    -o "${LOG_DIR}" \
    -v "REPO_ROOT=${REPO_ROOT},MANIFEST_PATH=${FULL_MANIFEST},RESULTS_DIR=${FULL_RESULTS},ANALYSIS_DIR=${ANALYSIS_DIR},PYTHON_PACKAGE_DIR=${PYTHON_PACKAGE_DIR}" \
    "${REPO_ROOT}/benchmarks/qsub_analyse_full_timegating_corrected.pbs.sh")

printf 'R_DEPS_JOB=%s\n' "${R_DEPS_JOB}"
printf 'PREFLIGHT_JOB=%s\n' "${PREFLIGHT_JOB}"
printf 'FULL_JOBS=%s\n' "${FULL_JOBS[*]}"
printf 'ANALYSIS_JOB=%s\n' "${ANALYSIS_JOB}"
