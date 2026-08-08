#!/bin/bash

#PBS -N flowmop_fig2_r_deps
#PBS -P eu59
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l jobfs=10GB
#PBS -l storage=gdata/eu59+gdata/dk92
#PBS -j oe
#PBS -m abe
#PBS -l wd

set -euo pipefail

module use /g/data/dk92/apps/Modules/modulefiles/
module load NCI-data-analysis

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
R_PACKAGE_DIR="${R_PACKAGE_DIR:?R_PACKAGE_DIR is required}"
R_SOURCE_REPO="${R_SOURCE_REPO:-}"

mkdir -p "${R_PACKAGE_DIR}"
export R_LIBS_USER="${R_PACKAGE_DIR}"

INSTALL_ARGS=("${R_PACKAGE_DIR}")
if [[ -n "${R_SOURCE_REPO}" ]]; then
    INSTALL_ARGS+=("${R_SOURCE_REPO}")
fi
Rscript "${REPO_ROOT}/benchmarks/install_full_timegating_r_dependencies.R" \
    "${INSTALL_ARGS[@]}"
Rscript "${REPO_ROOT}/benchmarks/check_full_timegating_environment.R"
