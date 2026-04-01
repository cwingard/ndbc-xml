#!/usr/bin/env bash
# =============================================================================
# ndbc.sh — Run the NDBC XML generation pipeline for a single OOI mooring.
#
# Usage:
#   ndbc.sh <SITE> <DEPLOYMENT_DIR> [extra args passed to ndbc_xml.ndbc]
#
# Examples:
#   ndbc.sh CE02 /home/ooiuser/data/parsed/ce02shsm/D00020
#   ndbc.sh CE02 /home/ooiuser/data/parsed/ce02shsm/D00020 --reprocess
#
# Crontab (one line per mooring, runs hourly):
#   0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE02 /home/ooiuser/data/parsed/ce02shsm/D00020
#   0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE04 /home/ooiuser/data/parsed/ce04ossm/D00019
#   0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE07 /home/ooiuser/data/parsed/ce07shsm/D00020
#   0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE09 /home/ooiuser/data/parsed/ce09ossm/D00020
#
# Exit codes mirror the Python pipeline:
#   0  — success, XML written
#   1  — error
#   2  — no new data (not an error; cron will not alert)
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Path configuration — adjust these for the deployment environment
# -----------------------------------------------------------------------------
# Full path to the Python interpreter in the ooi conda environment.
# Using the explicit path avoids relying on conda activation in non-interactive
# (cron) shells, where PATH is minimal and 'conda activate' is unavailable.
PYTHON=/home/ooiuser/miniforge3/envs/ooi/bin/python

# Root of the ndbc-xml package (directory containing ndbc_xml/).
PACKAGE_DIR=/home/ooiuser/code/ndbc-xml

# Directory for pipeline log files (one per site).
LOG_DIR=/home/ooiuser/logs/ndbc-xml

# -----------------------------------------------------------------------------
# Argument handling
# -----------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $(basename "$0") <SITE> <DEPLOYMENT_DIR> [extra args]" >&2
    exit 1
fi
SITE="$1"
DEPLOYMENT_DIR="$2"
shift 2
# Any remaining arguments (e.g. --reprocess, --alpha-date) are forwarded as-is.
EXTRA_ARGS=("$@")

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
# Ensure the log directory exists.
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SITE}.log"

# Put the package on PYTHONPATH so the module is importable even if the
# package has not been pip-installed in the environment.
export PYTHONPATH="${PACKAGE_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Run the pipeline
# -----------------------------------------------------------------------------
exec "${PYTHON}" -m ndbc_xml.ndbc \
    "${SITE}" \
    "${DEPLOYMENT_DIR}" \
    --log-file "${LOG_FILE}" \
    "${EXTRA_ARGS[@]}"
