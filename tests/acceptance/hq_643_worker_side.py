#!/usr/bin/env python3
"""hq#643: one real job, as the worker's account, in a scratch folder. Run by hq_643_job.py.

    sudo -n -u cyclops-worker python3 hq_643_worker_side.py <path to worker.py> <root> <id> <request>

The real run_job, the real Claude Code and token, and the account's real firewall. Not the unit's
sandbox: that needs the service. The one change: it doesn't clear the account when the job ends,
because that kills everything the account runs, and the installed service is on it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
sys.modules["worker"] = worker
spec.loader.exec_module(worker)

root, job_id, request = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
worker._clear_account = lambda: None
worker.prepare(root)
folder = root / worker.JOBS / job_id
(folder / worker.WORK).mkdir(parents=True)
(folder / worker.RUNNING).touch()
worker.run_job(
    root,
    job_id,
    request,
    claude="/opt/cyclops-worker/bin/claude",
    token_file=Path("/etc/cyclops-worker/claude-token"),
)
