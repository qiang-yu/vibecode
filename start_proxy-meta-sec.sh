#!/usr/bin/env bash

BACKEND_URL="http://localhost:19007/v1"   # backend LLM server base URL ending in /v1
SOURCE_ROLE="tool"                        # message role to rewrite
TARGET_ROLE="input"                       # replacement role value

LISTEN_HOST="localhost"
LISTEN_PORT=39007

REQUEST_TIMEOUT=600       # backend HTTP request timeout in seconds
LOG_LEVEL="info"          # debug | info | warning | error
LOG_FILE_NAME="proxy.log" # log file base name; runtime prepends YYYYMMDD_

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/proxy-meta-sec.py" \
    --backend-url   "${BACKEND_URL}" \
    --source-role   "${SOURCE_ROLE}" \
    --target-role   "${TARGET_ROLE}" \
    --host          "${LISTEN_HOST}" \
    --port          "${LISTEN_PORT}" \
    --timeout       "${REQUEST_TIMEOUT}" \
    --log-level     "${LOG_LEVEL}" \
    --log-file-name "${LOG_FILE_NAME}"
