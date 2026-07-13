#!/bin/bash

PIDS=(76896 76897 76898 76899)

echo "开始监视进程: ${PIDS[@]}"

while true; do
    all_done=true
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo "所有进程已结束，开始执行后续命令..."
        break
    fi
#   echo "monitor sleep 1s"
    sleep 1
done

python reason_tool_call_security.py

