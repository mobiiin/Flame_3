#!/usr/bin/env python3
"""Rename result artifacts in-place under ./results.

This script renames image files and .txt files inside run folders so names include:
- run/model identifier (folder name)
- fusion mode
- epoch count

It never moves or deletes files. By default it runs in preview mode.
Use --apply to perform renames.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_EXTS = {".txt"}
EPOCH_RE = re.compile(r"(?i)(\d+)\s*epoch")
FUSION_RE = re.compile(r"^\s*FUSION_MODE:\s*([A-Za-z_]+)\s*$", re.MULTILINE)
FINAL_EPOCHS_RE = re.compile(r"^\s*Final epochs:\s*(\d+)\s*$", re.MULTILINE)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "artifact"


def is_multimodal_run(run_name: str) -> bool:
    run_name = run_name.lower()
    return "__rgbt" in run_name or "__rgb_tiff" in run_name


def infer_epoch(run_dir: Path) -> str:
    for ancestor in [run_dir, *run_dir.parents]:
        match = EPOCH_RE.search(ancestor.name)
        if match:
            return match.group(1)

    log_path = run_dir.parent / "logs" / f"{run_dir.name}.log"
    if log_path.is_file():
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            match = FINAL_EPOCHS_RE.search(text)
            if match:
                return match.group(1)
        except OSError:
            pass

    return "na"


def infer_fusion_mode(run_dir: Path) -> str:
    txt_files = sorted(run_dir.glob("*.txt"))
    for txt in txt_files:
        try:
            text = txt.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = FUSION_RE.search(text)
        if match:
            return match.group(1).upper()

    lower_path = str(run_dir).lower()
    if "latefusion" in lower_path:
        return "LATE"
    if "earlyfusion" in lower_path or "ablation_suite_early" in lower_path:
        return "EARLY"
    if is_multimodal_run(run_dir.name):
        return "EARLY"
    return "NA"


def artifact_tag(file_path: Path) -> str:
    stem = file_path.stem.lower()
    if "confusion" in stem:
        return "confusion_matrix"
    if "learning" in stem:
        return "learning_curves"
    if "roc" in stem:
        return "roc_curve"
    if "final_results" in stem:
        return "final_results_log"
    return slugify(file_path.stem)


def build_new_name(run_name: str, fusion_mode: str, epoch: str, tag: str, suffix: str) -> str:
    return (
        f"model-{slugify(run_name)}__fusion-{slugify(fusion_mode)}"
        f"__epoch-{slugify(epoch)}__{tag}{suffix.lower()}"
    )


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name == "logs":
        return False
    names = {p.name for p in path.iterdir() if p.is_file()}
    has_model = "best_model.pt" in names
    has_target = any(Path(n).suffix.lower() in (IMAGE_EXTS | TEXT_EXTS) for n in names)
    return has_model and has_target


def target_files(run_dir: Path) -> Iterable[Path]:
    for p in sorted(run_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in TEXT_EXTS:
            yield p


def plan_renames(run_dir: Path) -> List[Tuple[Path, Path]]:
    run_name = run_dir.name
    fusion_mode = infer_fusion_mode(run_dir)
    epoch = infer_epoch(run_dir)
    ops: List[Tuple[Path, Path]] = []

    for src in target_files(run_dir):
        tag = artifact_tag(src)
        desired_name = build_new_name(run_name, fusion_mode, epoch, tag, src.suffix)
        dst = run_dir / desired_name
        if src == dst:
            continue

        if dst.exists() and dst != src:
            base = dst.stem
            suffix = dst.suffix
            counter = 2
            while True:
                candidate = run_dir / f"{base}__v{counter}{suffix}"
                if not candidate.exists() or candidate == src:
                    dst = candidate
                    break
                counter += 1

        ops.append((src, dst))
    return ops


def collect_run_dirs(root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for path in sorted(root.rglob("*")):
        if is_run_dir(path):
            run_dirs.append(path)
    return run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename result artifacts under ./results.")
    parser.add_argument("--root", default="./results", help="Results root directory (default: ./results)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames. Without this flag the script only prints a preview.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    run_dirs = collect_run_dirs(root)
    if not run_dirs:
        print(f"No run folders found under: {root}")
        return 0

    total_ops = 0
    for run_dir in run_dirs:
        ops = plan_renames(run_dir)
        if not ops:
            continue
        print(f"\n[{run_dir}]")
        for src, dst in ops:
            print(f"  {src.name} -> {dst.name}")
        total_ops += len(ops)

        if args.apply:
            for src, dst in ops:
                src.rename(dst)

    if total_ops == 0:
        print("No files required renaming.")
    else:
        mode = "APPLIED" if args.apply else "PREVIEW ONLY"
        print(f"\n{mode}: {total_ops} rename operation(s).")
        if not args.apply:
            print("Run with --apply to execute these renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
