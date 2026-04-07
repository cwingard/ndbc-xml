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
# Full path to the Python interpreter in the ndbc-xml conda environment.
# Using the explicit path avoids relying on conda activation in non-interactive
# (cron) shells, where PATH is minimal and 'conda activate' is unavailable.
PYTHON=/home/ooiuser/miniforge3/envs/ndbc-xml/bin/python

# Root of the ndbc-xml package (directory containing ndbc_xml/).
PACKAGE_DIR=/home/ooiuser/code/ndbc-xml

# Directory for pipeline log files (one per site).
LOG_DIR=/home/ooiuser/logs/ndbc-xml

# SFTP configuration for NDBC file delivery.
SFTP_KEY=/home/ooiuser/.ssh/ndbc
SFTP_USER=ooi.osu_sftp
SFTP_HOST=comms.ndbc.noaa.gov
SFTP_REMOTE_DIR=uploads

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
# Site metadata
# -----------------------------------------------------------------------------
case "${SITE}" in
    CE02) NDBC_ID=46097 ;;
    CE04) NDBC_ID=46098 ;;
    CE07) NDBC_ID=46099 ;;
    CE09) NDBC_ID=46100 ;;
    *)
        echo "Unknown site: ${SITE}" >&2
        exit 1
        ;;
esac

# XML output directory — matches the pipeline's default (no --xml-out needed).
XML_OUT_DIR="${DEPLOYMENT_DIR}/buoy/metbk/xml"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
# Ensure the log directory exists (the pipeline creates XML_OUT_DIR itself).
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SITE}.log"

# Put the package on PYTHONPATH so the module is importable even if the
# package has not been pip-installed in the environment.
export PYTHONPATH="${PACKAGE_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Run the pipeline
# -----------------------------------------------------------------------------
# Check now (before shifting args) whether this is a reprocess run so we can
# skip SFTP after the pipeline finishes.
REPROCESS=0
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    [[ "${arg}" == "--reprocess" ]] && REPROCESS=1 && break
done

# Touch a temp file immediately before the pipeline runs so we can identify
# which XML files were written during this run (handles both normal and
# --reprocess cases, where multiple files may be written).
RUN_MARKER=$(mktemp)
trap 'rm -f "${RUN_MARKER}"' EXIT

EXIT_CODE=0
"${PYTHON}" -m ndbc_xml.ndbc \
    "${SITE}" \
    "${DEPLOYMENT_DIR}" \
    --log-file "${LOG_FILE}" \
    "${EXTRA_ARGS[@]}" || EXIT_CODE=$?

# -----------------------------------------------------------------------------
# SFTP transfer — only when the pipeline wrote new XML (exit 0) and this is
# not a reprocess run (NDBC may not accept back-filled files).
# -----------------------------------------------------------------------------
if [[ ${EXIT_CODE} -eq 0 && ${REPROCESS} -eq 0 ]]; then
    # Collect all XML files for this station written during this run.
    WRITTEN=()
    while IFS= read -r -d '' f; do
        WRITTEN+=("$f")
    done < <(find "${XML_OUT_DIR}" -name "*-${NDBC_ID}.xml" -newer "${RUN_MARKER}" -print0)

    if [[ ${#WRITTEN[@]} -eq 0 ]]; then
        echo "Warning: pipeline succeeded but no new XML files found in ${XML_OUT_DIR}" >&2
    else
        for XML_FILE in "${WRITTEN[@]}"; do
            FILENAME=$(basename "${XML_FILE}")
            sftp -i "${SFTP_KEY}" "${SFTP_USER}@${SFTP_HOST}" <<EOF
cd ${SFTP_REMOTE_DIR}
put ${XML_FILE} ${FILENAME}
bye
EOF
        done
    fi
fi

exit ${EXIT_CODE}
