"""Move all result files whose "error" field is not empty out of the log directory,
so that re-running the benchmark will automatically re-run those failed tasks.
Result files that are empty or not valid JSON are treated as failed and moved too
(modified by qiangyu).

The directory hierarchy is preserved inside the backup directory, e.g.
    <logdir>/vllm_parsed-drift/slack/user_task_0/important_instructions/injection_task_1.json
is moved to
    <logdir>_error_backup/vllm_parsed-drift/slack/user_task_0/important_instructions/injection_task_1.json

Usage:
    python util_scripts/move_error_results.py <logdir>            # list only, do not move
    python util_scripts/move_error_results.py <logdir> --move     # move to <logdir>_error_backup
"""

import json
import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

logdir = Path(sys.argv[1]).resolve()
do_move = "--move" in sys.argv
backup = logdir.parent / f"{logdir.name}_error_backup"

# Modified by qiangyu: number of path parts of a task result file relative to logdir,
# i.e. <pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json
RESULT_PATH_DEPTH = 5

found = 0
for f in sorted(logdir.rglob("*.json")):
    rel = f.relative_to(logdir)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        # Modified by qiangyu: previously unparseable files were silently skipped, so an empty or
        # truncated result JSON (left behind when the benchmark is killed while saving) stayed in
        # logdir and was never re-run by this script. Treat it as a failed task and move it out too.
        # Only files at the task result location are moved, so unrelated JSON files are left alone.
        if len(rel.parts) != RESULT_PATH_DEPTH:
            continue
        reason = f"<empty or invalid JSON: {e}>"
    else:
        # Only handle task result files
        if not isinstance(data, dict) or "user_task_id" not in data:
            continue
        if not data.get("error"):
            continue
        reason = str(data["error"])

    # Modified by qiangyu: shared reporting/moving logic for both error results and corrupted files
    found += 1
    print(f"{rel}  ->  {reason[:80]}")
    if do_move:
        # Keep the same relative path under the backup directory
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))

if do_move:
    print(f"\nTotal {found} failed task(s), moved to {backup}")
else:
    print(f"\nTotal {found} failed task(s) (not moved, add --move to move them)")
