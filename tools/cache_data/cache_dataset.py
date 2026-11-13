"""
cache_dataset.py
----------------
Pre-processes every mesh and saves the config-encoded .pt cache next to each
OBJ, so training / inference never process meshes on the fly.

Usage
-----
    # From the run config (caches data.train_csv + data.test_csv)
    python tools/cache_data/cache_dataset.py --config configs/mae_v3.yml

    # CLI flags override the YAML, or work without it
    python tools/cache_data/cache_dataset.py --csv caption_breakingbad_3.csv --root . \
        --n_fps 768 --disk_grid 32 --disk_radius_frac 0.07 --curv_knn 30 --workers 4

    # Reprocess everything
    python tools/cache_data/cache_dataset.py --config configs/mae_v3.yml --force

Parallelism (why --workers > 1 used to be slower than serial)
-------------------------------------------------------------
Each mesh is dominated by NumPy/SciPy linear algebra (quadric fits, one RBF
solve per disk), open3d (OpenMP) and torch, and each of those spawns one
thread per core by default. N workers x all-cores threads oversubscribes the
CPU. Workers are now:
  * started with "spawn" after OMP/MKL/OpenBLAS thread caps are put in the
    environment (forked workers inherit the parent's already-initialised
    thread pools, so the caps never applied, and fork + OpenMP can hang);
  * capped to --threads_per_worker (default: cores // workers);
  * recycled every --max_tasks_per_child meshes (Python >= 3.11), so open3d /
    SciPy allocator growth can't accumulate;
  * fed a bounded queue; a worker crash (segfault in open3d, OOM-kill) only
    retries the meshes that were in flight instead of killing the run;
  * terminated on Ctrl-C (no orphan processes).
Every cache is written to a temp file and renamed, so an interrupted run never
leaves a truncated .pt that the next run would "skip".
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"          # BreakingBadDataset.py + run_config.py
for _p in (SCRIPTS, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from run_config import log_resolved, parse_with_config  # noqa: E402

log = logging.getLogger(__name__)

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-cache Breaking Bad dataset .pt files")
    p.add_argument("--csv", nargs="+", required=True, help="One or more caption CSVs")
    p.add_argument("--root", required=True)

    # Geometry — same keys/defaults as train_mae.py / check_cache.py
    p.add_argument("--n_fps",            type=int,   default=512)
    p.add_argument("--disk_grid",        type=int,   default=16)
    p.add_argument("--disk_radius_frac", type=float, default=0.07)
    p.add_argument("--disk_fill",        default="rbf", choices=["rbf", "nearest", "zero"])
    p.add_argument("--curv_knn",         type=int,   default=30)
    p.add_argument("--fourier_bands",    type=int,   default=0)
    p.add_argument("--curv_channels",    type=int,   default=6, choices=[2, 3, 4, 6],
                   help="Accepted for command compatibility; has NO effect — the "
                        "cache always stores all 6 channels.")
    p.add_argument("--use_tangent_disk", action="store_false", default=True,
                   help="PASSING this flag = point mode (disk_grid forced to 1)")

    # Execution
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes. 0 or 1 = serial in this process.")
    p.add_argument("--threads_per_worker", type=int, default=0,
                   help="BLAS/OpenMP/torch threads per worker. 0 = cores // workers "
                        "(min 1). Serial mode keeps the library default unless set.")
    p.add_argument("--max_tasks_per_child", type=int, default=50,
                   help="Recycle each worker after this many meshes (Python >= 3.11). "
                        "0 = never.")
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if the cache file exists")
    p.add_argument("--skip_errors", action="store_true", default=True,
                   help="Continue past meshes that fail (default)")
    p.add_argument("--fail_fast", dest="skip_errors", action="store_false",
                   help="Stop at the first failing mesh")
    return p


def dataset_cfg_dict(args) -> dict:
    """Picklable DatasetConfig kwargs — identical to train_mae.py / infer_mae.py."""
    return dict(
        n_fps=args.n_fps, disk_grid=args.disk_grid,
        disk_radius_frac=args.disk_radius_frac, disk_channels=6,
        disk_fill=args.disk_fill, curv_knn=args.curv_knn, cache_processed=True,
        skip_parse_errors=True, fourier_bands=args.fourier_bands,
        use_tangent_disk=args.use_tangent_disk,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Worker side (top-level so "spawn" can pickle it)
# ──────────────────────────────────────────────────────────────────────────────

_THREAD_LIMITER = None   # keep threadpoolctl limiter alive for the worker's life


def limit_threads(n: int) -> None:
    """Cap BLAS/OpenMP/torch threads in THIS process."""
    global _THREAD_LIMITER
    for v in THREAD_VARS:
        os.environ[v] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMITER = threadpool_limits(limits=n)
    except ImportError:
        pass   # env vars still cover pools that initialise after this point
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


def _init_worker(threads: int) -> None:
    limit_threads(threads)
    # Import the heavy stack once per worker, after the thread caps are in place.
    import BreakingBadDataset  # noqa: F401


def _cache_one(job: tuple[str, dict, bool]) -> tuple[str, str, float]:
    """Build one mesh's cache. Returns (obj_path, status, seconds).
    status: 'cached' | 'skipped' | 'error: ...'"""
    obj_str, cfg_dict, force = job
    import torch
    from BreakingBadDataset import DatasetConfig, _cache_path, process_mesh

    cfg = DatasetConfig(**cfg_dict)
    obj = Path(obj_str)
    pt  = _cache_path(obj, cfg)
    if pt.exists() and not force:
        return obj_str, "skipped", 0.0

    for stale in pt.parent.glob(f"{pt.name}.tmp*"):   # left by a killed run
        stale.unlink(missing_ok=True)
    t0  = time.time()
    tmp = pt.with_name(f"{pt.name}.tmp{os.getpid()}")
    try:
        # cache_processed=False: process_mesh neither reads nor writes the cache;
        # we write it ourselves, atomically.
        result = process_mesh(obj, replace(cfg, cache_processed=False))
        torch.save(result, tmp)
        os.replace(tmp, pt)
        return obj_str, "cached", time.time() - t0
    except Exception as e:
        return obj_str, f"error: {type(e).__name__}: {e}", time.time() - t0
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# Job list
# ──────────────────────────────────────────────────────────────────────────────

def collect_jobs(csv_files: list[str], root: Path) -> tuple[list[str], list[str]]:
    """Unique existing OBJ paths, with the same row filtering as BreakingBadDataset."""
    seen, jobs, missing = set(), [], []
    for csv in csv_files:
        df = pd.read_csv(csv, sep=",", dtype=str)
        df.columns = df.columns.str.strip()
        if "parse_error" in df.columns:
            df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
        df = df[:-1]   # mirrors BreakingBadDataset's df[:-1]
        log.info(f"{csv}: {len(df)} rows, {df['path'].nunique()} meshes")
        for rel in df["path"]:
            obj = root / str(rel).strip()
            key = str(obj.resolve())
            if key in seen:
                continue
            seen.add(key)
            (jobs if obj.exists() else missing).append(str(obj))
    return jobs, missing


# ──────────────────────────────────────────────────────────────────────────────
# Progress bookkeeping
# ──────────────────────────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int, to_run: int, every: int = 50):
        self.total, self.to_run, self.every = total, to_run, every
        self.c = Counter()
        self.errors: list[tuple[str, str]] = []
        self.busy_s = 0.0
        self.t0 = time.time()

    def add(self, path: str, status: str, dt: float) -> None:
        kind = status.split(":", 1)[0]
        self.c[kind] += 1
        self.busy_s += dt
        if kind not in ("cached", "skipped"):
            self.errors.append((path, status))
            log.warning(f"{Path(path).parent.parent.name}/{Path(path).name}: {status}")
        done = sum(self.c.values())
        if done % self.every == 0 or done == self.total:
            self.report(done)

    def report(self, done: int) -> None:
        elapsed = time.time() - self.t0
        n_cached = self.c["cached"]
        rate = n_cached / max(elapsed, 1e-6)
        eta = (self.to_run - n_cached) / max(rate, 1e-6) if n_cached else float("nan")
        per_mesh = self.busy_s / max(n_cached, 1)
        log.info(f"[{done}/{self.total}]  cached={n_cached}  skipped={self.c['skipped']}  "
                 f"errors={len(self.errors)}  {rate:.2f} mesh/s  "
                 f"({per_mesh:.1f}s per mesh in-worker)  ETA={eta/60:.1f}min")


# ──────────────────────────────────────────────────────────────────────────────
# Runners
# ──────────────────────────────────────────────────────────────────────────────

def run_serial(job_args, prog: Progress, skip_errors: bool) -> None:
    for ja in job_args:
        path, status, dt = _cache_one(ja)
        prog.add(path, status, dt)
        if status.startswith("error") and not skip_errors:
            raise SystemExit(f"Failed: {path} → {status}")


def run_parallel(job_args, prog: Progress, workers: int, threads: int,
                 max_tasks_per_child: int, skip_errors: bool) -> None:
    # Children read these at import time — set before they are spawned.
    for v in THREAD_VARS:
        os.environ[v] = str(threads)

    kw = dict(max_workers=workers, mp_context=mp.get_context("spawn"),
              initializer=_init_worker, initargs=(threads,))
    if max_tasks_per_child > 0:
        if sys.version_info >= (3, 11):
            kw["max_tasks_per_child"] = max_tasks_per_child
        else:
            log.warning("max_tasks_per_child needs Python >= 3.11 — workers are not recycled")

    queue, crashes = deque(job_args), Counter()
    max_inflight = workers * 4
    while queue:
        pool, inflight = ProcessPoolExecutor(**kw), {}
        try:
            while queue or inflight:
                while queue and len(inflight) < max_inflight:
                    ja = queue.popleft()
                    inflight[pool.submit(_cache_one, ja)] = ja
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    path, status, dt = fut.result()     # BrokenProcessPool propagates
                    inflight.pop(fut)
                    prog.add(path, status, dt)
                    if status.startswith("error") and not skip_errors:
                        raise SystemExit(f"Failed: {path} → {status}")
            pool.shutdown(wait=True)
        except BrokenProcessPool:
            # A worker died hard. Every mesh still in flight is suspect; retry
            # each once, give up on meshes that were in flight for two crashes.
            lost = list(inflight.values())
            for ja in lost:
                crashes[ja[0]] += 1
                if crashes[ja[0]] < 2:
                    queue.appendleft(ja)
                else:
                    prog.add(ja[0], "error: worker process crashed twice on this mesh", 0.0)
            log.warning(f"A worker crashed (segfault / OOM-kill?). Restarting the pool; "
                        f"{sum(crashes[j[0]] < 2 for j in lost)} in-flight meshes re-queued.")
            _kill_pool(pool)
        except BaseException:
            log.warning("Stopping — terminating worker processes")
            _kill_pool(pool)
            raise


def _kill_pool(pool: ProcessPoolExecutor) -> None:
    for p in list(getattr(pool, "_processes", {}).values()):
        if p.is_alive():
            p.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_with_config(build_parser(), "cache_dataset", argv)
    log_resolved(args, ["csv", "root", "n_fps", "use_tangent_disk", "disk_grid",
                        "disk_radius_frac", "disk_fill", "curv_knn", "fourier_bands",
                        "workers", "threads_per_worker", "max_tasks_per_child", "force"])
    if "curv_channels" in args.cli_overrides:
        log.info("--curv_channels has no effect here: caches always hold all 6 channels.")

    from BreakingBadDataset import DatasetConfig, _cache_path
    cfg_dict = dataset_cfg_dict(args)
    cfg = DatasetConfig(**cfg_dict)

    root = Path(args.root)
    jobs, missing = collect_jobs(args.csv, root)
    log.info(f"Unique meshes: {len(jobs) + len(missing)}  missing OBJ: {len(missing)}")
    if missing:
        log.warning(f"First 5 missing: {missing[:5]}")
    if not jobs:
        log.info("Nothing to do.")
        return

    already = 0 if args.force else sum(_cache_path(Path(p), cfg).exists() for p in jobs)
    to_run = len(jobs) - already
    log.info(f"Cache filename : {_cache_path(Path(jobs[0]), cfg).name}")
    log.info(f"Already cached : {already}   will process: {to_run}")

    job_args = [(p, cfg_dict, args.force) for p in jobs]
    prog = Progress(total=len(jobs), to_run=to_run)

    cores = os.cpu_count() or 1
    if args.workers <= 1:
        if args.threads_per_worker > 0:
            limit_threads(args.threads_per_worker)
        log.info("Running serially in this process")
        run_serial(job_args, prog, args.skip_errors)
    else:
        threads = args.threads_per_worker or max(1, cores // args.workers)
        if args.workers * threads > cores:
            log.warning(f"{args.workers} workers × {threads} threads > {cores} cores — "
                        f"expect oversubscription")
        log.info(f"Running {args.workers} workers × {threads} thread(s) on {cores} cores "
                 f"(spawn, recycle every {args.max_tasks_per_child or '∞'} meshes)")
        run_parallel(job_args, prog, args.workers, threads,
                     args.max_tasks_per_child, args.skip_errors)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - prog.t0
    log.info("─" * 50)
    log.info(f"Done in {elapsed/60:.1f} min  cached={prog.c['cached']}  "
             f"skipped={prog.c['skipped']}  errors={len(prog.errors)}")
    if prog.c["cached"]:
        log.info(f"Parallel efficiency: {prog.busy_s / max(elapsed, 1e-6):.1f} meshes "
                 f"processed concurrently on average (ideal ≈ {max(args.workers, 1)})")
    if prog.errors:
        err_path = root / "cache_errors.txt"
        err_path.write_text("".join(f"{p}\t{e}\n" for p, e in prog.errors))
        log.warning(f"{len(prog.errors)} failed meshes → {err_path}")

    have = sum(_cache_path(Path(p), cfg).exists() for p in jobs)
    log.info(f"Cache completeness: {have}/{len(jobs)} meshes have .pt files")
    if have < len(jobs):
        log.warning(f"{len(jobs) - have} meshes still missing — check cache_errors.txt and re-run")


if __name__ == "__main__":
    main()
