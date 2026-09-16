"""Run directories, per-instance JSONL logs, and the global summary CSV."""
from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SUMMARY_COLUMNS = [
    "timestamp", "run_id", "problem", "network", "n_nodes", "n_edges", "method", "beta", "sigma", "T", "nu",
    "train_sims", "test_sims", "test_sigma", "seed", "n_instances", "avg_outbreak",
    "top1", "top1_se", "top3", "top5", "ed", "rr", "css", "brier", "resist",
    "fit_time_s", "infer_time_per_instance_s", "sim_time_s", "git_commit", "notes",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid() % 10000:04d}"


def setup_logger(run_dir: str, name: str = "trace_e") -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class RunLogger:
    """Owns one run directory under ``results/`` and the shared summary CSV."""

    def __init__(self, problem: str, tag: str, results_dir: str = RESULTS_DIR, config: dict | None = None):
        self.problem = problem
        self.run_id = make_run_id(f"{problem}_{tag}")
        self.results_dir = results_dir
        self.run_dir = os.path.join(results_dir, "runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log = setup_logger(self.run_dir)
        self.summary_path = os.path.join(results_dir, f"summary_{problem}.csv")
        self.t0 = time.time()
        cfg = dict(config or {})
        cfg.update({"run_id": self.run_id, "git_commit": git_commit(), "python": platform.python_version(),
                    "argv": sys.argv, "started": datetime.now(timezone.utc).isoformat()})
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        self.log.info("run %s started, dir=%s", self.run_id, self.run_dir)

    def instance_writer(self, method: str):
        path = os.path.join(self.run_dir, f"instances_{method}.jsonl.gz")
        return gzip.open(path, "wt")

    def write_summary(self, row: dict):
        row = {k: row.get(k, "") for k in SUMMARY_COLUMNS}
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["run_id"] = self.run_id
        row["problem"] = self.problem
        row["git_commit"] = row.get("git_commit") or git_commit()
        new = not os.path.exists(self.summary_path)
        with open(self.summary_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        with open(os.path.join(self.run_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def finish(self):
        self.log.info("run %s finished in %.1fs", self.run_id, time.time() - self.t0)


def append_csv(path: str, row: dict, columns: list[str]):
    """Append one row to a CSV with a fixed column set (header written on first use)."""
    row = {k: row.get(k, "") for k in columns}
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != list(columns):  # schema changed: migrate existing rows to the new column set
            with open(path, newline="") as f:
                old = list(csv.DictReader(f))
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                w.writeheader()
                for r in old:
                    w.writerow({k: r.get(k, "") for k in columns})
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow(row)
