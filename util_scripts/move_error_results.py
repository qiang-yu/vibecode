"""Move all result files whose "error" field is not empty out of the log directory,
so that re-running the benchmark will automatically re-run those failed tasks.

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

found = 0
for f in sorted(logdir.rglob("*.json")):
    try:
        data = json.loads(f.read_text())
    except Exception:
        continue
    # Only handle task result files
    if not isinstance(data, dict) or "user_task_id" not in data:
        continue
    if data.get("error"):
        found += 1
        rel = f.relative_to(logdir)
        print(f"{rel}  ->  {str(data['error'])[:80]}")
        if do_move:
            # Keep the same relative path under the backup directory
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))

if do_move:
    print(f"\nTotal {found} failed task(s), moved to {backup}")
else:
    print(f"\nTotal {found} failed task(s) (not moved, add --move to move them)")
