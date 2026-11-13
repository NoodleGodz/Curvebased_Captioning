"""
check_cache.py — validate per-config .pt cache files against the run config.

The cache filename and expected shapes come from BreakingBadDataset itself
(_cache_path + DatasetConfig), so this cannot drift from what training and
inference load.

Modes (--check_mode)
--------------------
  exists  only checks the file is there                       (seconds)
  header  DEFAULT. Opens each file memory-mapped and checks keys + shapes
          without reading tensor data; a truncated/broken file still fails
          because the zip directory sits at the end of the file.  (fast)
  full    reads every tensor and also checks for NaN/Inf.      (slow, ~25 MB/file
          at n_fps=1024, D=32)

Files are checked in parallel threads (--check_workers); the work is I/O.

Usage
-----
    python tools/cache_data/check_cache.py --config configs/mae_v3.yml                      # header mode
    python tools/cache_data/check_cache.py --config configs/mae_v3.yml --check_mode full    # deep check
    python tools/cache_data/check_cache.py --config configs/mae_v3.yml --delete_bad         # then rebuild
    python tools/cache_data/check_cache.py --csv train.csv --root . --n_fps 768 --disk_grid 32
"""

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # threads here are for I/O, not math
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))   # BreakingBadDataset, run_config
from run_config import log_resolved, parse_with_config  # noqa: E402
from BreakingBadDataset import DatasetConfig, _cache_path  # noqa: E402

log = logging.getLogger(__name__)
CACHE_KEYS = ("pos", "norm", "curv", "data", "pos_enc")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate per-config .pt cache files")
    p.add_argument("--csv", nargs="+", required=True, help="One or more caption CSVs")
    p.add_argument("--root", default=".")

    # Geometry — same keys/defaults as train_mae.py / cache_dataset.py
    p.add_argument("--n_fps",            type=int,   default=512)
    p.add_argument("--disk_grid",        type=int,   default=16)
    p.add_argument("--disk_radius_frac", type=float, default=0.07)
    p.add_argument("--disk_fill",        default="rbf", choices=["rbf", "nearest", "zero"])
    p.add_argument("--curv_knn",         type=int,   default=30)
    p.add_argument("--fourier_bands",    type=int,   default=0)
    p.add_argument("--use_tangent_disk", action="store_false", default=True,
                   help="PASSING this flag = point mode (disk_grid forced to 1)")

    p.add_argument("--check_mode", default="header", choices=["exists", "header", "full"])
    p.add_argument("--check_workers", type=int, default=16,
                   help="Parallel I/O threads (use fewer on a spinning disk / NFS)")
    p.add_argument("--delete_bad", action="store_true",
                   help="Delete corrupt / wrong-shape / non-finite files so they get rebuilt")
    p.add_argument("--quiet", action="store_true", help="Only print summary")
    return p


def dataset_config(args) -> DatasetConfig:
    """Exactly the DatasetConfig train_mae.py / infer_mae.py / cache_dataset.py build."""
    return DatasetConfig(
        n_fps=args.n_fps, disk_grid=args.disk_grid,
        disk_radius_frac=args.disk_radius_frac, disk_channels=6,
        disk_fill=args.disk_fill, curv_knn=args.curv_knn, cache_processed=True,
        fourier_bands=args.fourier_bands, use_tangent_disk=args.use_tangent_disk,
    )


def expected_shapes(cfg: DatasetConfig) -> dict:
    """Shapes process_mesh writes (always all 6 curvature channels)."""
    pos_enc_dim = 6 + (2 * cfg.fourier_bands * 3 if cfg.fourier_bands > 0 else 0)
    D = cfg.disk_grid if cfg.use_tangent_disk else 1
    return {"pos": (cfg.n_fps, 3), "norm": (cfg.n_fps, 3), "curv": (cfg.n_fps, 6),
            "data": (cfg.n_fps, 6, D, D), "pos_enc": (cfg.n_fps, pos_enc_dim)}


def load_paths(csv_files: list[str], root: Path) -> list[Path]:
    """Same row filtering as BreakingBadDataset (parse_error rows, last row)."""
    paths = set()
    for csv in csv_files:
        df = pd.read_csv(csv, dtype=str, usecols=lambda c: c.strip() in ("path", "parse_error"))
        df.columns = df.columns.str.strip()
        if "parse_error" in df.columns:
            df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
        df = df[:-1]   # mirrors BreakingBadDataset's df[:-1]
        paths |= {root / p.strip() for p in df["path"]}
        log.info(f"{csv}: {df['path'].nunique()} meshes")
    return sorted(paths)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file check
# ──────────────────────────────────────────────────────────────────────────────

def _load(pt: Path, mode: str):
    if mode == "header":
        try:   # torch >= 2.1, zipfile format: tensor data is NOT read
            return torch.load(pt, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            pass   # old torch / legacy format — a full load decides if it's corrupt
    return torch.load(pt, map_location="cpu", weights_only=False)


def check_one(obj: Path, cfg: DatasetConfig, exp: dict, mode: str):
    """→ (obj, status, detail, pt). status: ok | missing | corrupt | wrong_shape | nonfinite"""
    pt = _cache_path(obj, cfg)
    if not pt.is_file():
        return obj, "missing", f"expected {pt.name}", pt
    if mode == "exists":
        return obj, "ok", "", pt
    try:
        cache = _load(pt, mode)
        if not isinstance(cache, dict):
            raise TypeError(f"not a dict ({type(cache).__name__})")
    except Exception as e:
        return obj, "corrupt", str(e)[:100], pt

    problems = []
    for key, shape in exp.items():
        if key not in cache:
            problems.append(f"missing key '{key}'")
        elif tuple(cache[key].shape) != shape:
            problems.append(f"{key}={tuple(cache[key].shape)} expected {shape}")
    if problems:
        return obj, "wrong_shape", "; ".join(problems), pt

    if mode == "full":
        bad = [k for k in CACHE_KEYS if not torch.isfinite(cache[k]).all()]
        if bad:
            return obj, "nonfinite", f"NaN/Inf in {bad}", pt
    return obj, "ok", "", pt


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_with_config(build_parser(), "check_cache", argv)
    log_resolved(args, ["csv", "root", "n_fps", "use_tangent_disk", "disk_grid",
                        "disk_radius_frac", "disk_fill", "curv_knn", "fourier_bands",
                        "check_mode", "check_workers", "delete_bad"])

    cfg = dataset_config(args)
    exp = expected_shapes(cfg)
    paths = load_paths(args.csv, Path(args.root))
    total = len(paths)
    log.info(f"Cache filename pattern: {_cache_path(Path('piece_0.obj'), cfg).name}")
    log.info("Expected shapes: " + "  ".join(f"{k}={s}" for k, s in exp.items()))

    counts = dict(ok=0, missing=0, corrupt=0, wrong_shape=0, nonfinite=0)
    bad, lock = [], threading.Lock()
    t0 = last = time.time()

    def record(res):
        nonlocal last
        obj, status, detail, pt = res
        with lock:
            counts[status] += 1
            if status != "ok":
                bad.append((str(obj), status, detail))
                if args.delete_bad and status != "missing":
                    pt.unlink(missing_ok=True)
            done = sum(counts.values())
            now = time.time()
            if not args.quiet and (now - last > 1.0 or done == total):
                last = now
                rate = done / max(now - t0, 1e-6)
                print(f"\r[{done}/{total}]  " + "  ".join(f"{k}={v}" for k, v in counts.items())
                      + f"  {rate:.0f} files/s   ", end="", flush=True)

    workers = max(1, args.check_workers)
    if workers == 1:
        for obj in paths:
            record(check_one(obj, cfg, exp, args.check_mode))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(lambda o: check_one(o, cfg, exp, args.check_mode), paths,
                                chunksize=1):
                record(res)

    elapsed = time.time() - t0
    print()
    print("─" * 56)
    print(f"Mode         : {args.check_mode}   ({workers} threads, {elapsed:.1f}s, "
          f"{total / max(elapsed, 1e-6):.0f} files/s)")
    print(f"Total        : {total}")
    for k, v in counts.items():
        print(f"{k:<13}: {v}" + (f"  ({v / max(total, 1) * 100:.1f}%)" if k == "ok" else ""))
    print("─" * 56)

    if not bad:
        print("All caches valid ✓")
        sys.exit(0)

    bad.sort()
    if not args.quiet:
        print(f"\nBad files ({len(bad)}):")
        for obj_path, status, detail in bad[:30]:
            print(f"  {Path(obj_path).parent.parent.name}/{Path(obj_path).name:<16} "
                  f"{status:<12} {detail}")
        if len(bad) > 30:
            print(f"  ... and {len(bad) - 30} more")
    out = Path("bad_cache.txt")
    out.write_text("".join(f"{p}\t{s}\t{d}\n" for p, s, d in bad))
    print(f"\nFull list → {out}")
    if args.delete_bad:
        print("Bad files deleted (missing ones untouched). Rebuild with:")
    else:
        print("Delete corrupt/wrong-shape files with --delete_bad, then rebuild with:")
    print("  python tools/cache_data/cache_dataset.py --config <same yml>")
    sys.exit(1)


if __name__ == "__main__":
    main()
