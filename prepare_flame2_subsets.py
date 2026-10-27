#!/usr/bin/env python3
"""
Prepare FLAME2 Fire/No Fire subsets from mixed folders.

Behavior:
- Reads images directly inside the two mixed source directories:
  - Flame 2/Original Sized RGB Images
  - Flame 2/Original Sized Thermal Images
- Splits by filename index rule:
  - 1..13700   -> No Fire
  - 13701..53451 -> Fire
- Copies into label subfolders under each source directory.
- Re-encodes copied files to JPEG under a target file size (default 100 KB)
  while preserving aspect ratio.
- Never deletes or modifies original source files.

Example:
  python prepare_flame2_subsets.py --base-data-dir /home/mhabibp/Downloads
"""

from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_BASE_DATA_DIR = "/home/mhabibp/Downloads"

NO_FIRE_END = 13700
MAX_INDEX = 53451
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class ProcessResult:
    source_dir: Path
    total_top_level_images: int
    copied: int
    skipped_existing: int
    skipped_bad_name: int
    skipped_out_of_range: int
    failed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split FLAME2 mixed images into Fire/No Fire folders.")
    parser.add_argument("--base-data-dir", default=DEFAULT_BASE_DATA_DIR, help="Base directory containing 'Flame 2'.")
    parser.add_argument(
        "--target-size-kb",
        type=int,
        default=100,
        help="Target max size per copied image in KB (default: 100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would happen; do not write files.",
    )
    return parser.parse_args()


def extract_last_integer(stem: str) -> Optional[int]:
    """Extract the last integer group from a filename stem."""
    match = re.search(r"(\d+)(?!.*\d)", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def label_from_index(index: int) -> Optional[str]:
    if 1 <= index <= NO_FIRE_END:
        return "No Fire"
    if NO_FIRE_END + 1 <= index <= MAX_INDEX:
        return "Fire"
    return None


def ensure_output_dirs(source_dir: Path, dry_run: bool) -> Tuple[Path, Path]:
    fire_dir = source_dir / "Fire"
    no_fire_dir = source_dir / "No Fire"
    if not dry_run:
        fire_dir.mkdir(parents=True, exist_ok=True)
        no_fire_dir.mkdir(parents=True, exist_ok=True)
    return fire_dir, no_fire_dir


def iter_top_level_images(source_dir: Path) -> Iterable[Path]:
    for p in sorted(source_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def encode_under_target_jpeg(
    img: Image.Image,
    target_bytes: int,
    min_side_px: int = 320,
) -> bytes:
    """
    Encode image as JPEG under target_bytes, preserving aspect ratio.

    Strategy:
    - Try multiple scales, then multiple quality levels.
    - Keep aspect ratio at each scale.
    - Return best effort bytes if strict target cannot be met.
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # Keep grayscale as L for thermal if possible.
        pass

    width, height = img.size
    longest = max(width, height)

    scale_steps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    quality_steps = [90, 82, 74, 66, 58, 50, 42, 35, 28, 22]

    best_bytes: Optional[bytes] = None

    for scale in scale_steps:
        new_w = max(int(width * scale), min_side_px)
        new_h = max(int(height * scale), min_side_px)

        # Do not upscale smaller originals.
        if new_w > width or new_h > height:
            resized = img
        else:
            resized = img.resize((new_w, new_h), Image.LANCZOS)

        for quality in quality_steps:
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
            data = buf.getvalue()

            if best_bytes is None or len(data) < len(best_bytes):
                best_bytes = data

            if len(data) <= target_bytes:
                return data

        if longest * scale <= min_side_px:
            break

    # Best effort fallback if strict limit cannot be reached.
    if best_bytes is None:
        raise RuntimeError("Failed to encode image.")
    return best_bytes


def process_source_dir(
    source_dir: Path,
    target_size_kb: int,
    overwrite: bool,
    dry_run: bool,
) -> ProcessResult:
    fire_dir, no_fire_dir = ensure_output_dirs(source_dir, dry_run=dry_run)

    total_top_level_images = 0
    copied = 0
    skipped_existing = 0
    skipped_bad_name = 0
    skipped_out_of_range = 0
    failed = 0

    target_bytes = target_size_kb * 1024

    for src in iter_top_level_images(source_dir):
        total_top_level_images += 1

        idx = extract_last_integer(src.stem)
        if idx is None:
            skipped_bad_name += 1
            continue

        label = label_from_index(idx)
        if label is None:
            skipped_out_of_range += 1
            continue

        dst_dir = fire_dir if label == "Fire" else no_fire_dir
        dst_name = f"{src.stem}.jpg"
        dst = dst_dir / dst_name

        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue

        if dry_run:
            copied += 1
            continue

        try:
            with Image.open(src) as img:
                encoded = encode_under_target_jpeg(img, target_bytes=target_bytes)
            with open(dst, "wb") as f:
                f.write(encoded)
            copied += 1
        except Exception:
            failed += 1

    return ProcessResult(
        source_dir=source_dir,
        total_top_level_images=total_top_level_images,
        copied=copied,
        skipped_existing=skipped_existing,
        skipped_bad_name=skipped_bad_name,
        skipped_out_of_range=skipped_out_of_range,
        failed=failed,
    )


def main() -> int:
    args = parse_args()

    base = Path(args.base_data_dir)
    rgb_dir = base / "Flame 2" / "Original Sized RGB Images"
    thermal_dir = base / "Flame 2" / "Original Sized Thermal Images"

    if not rgb_dir.is_dir():
        print(f"ERROR: RGB source directory not found: {rgb_dir}")
        return 1
    if not thermal_dir.is_dir():
        print(f"ERROR: Thermal source directory not found: {thermal_dir}")
        return 1

    print("Preparing FLAME2 subsets")
    print(f"Base data dir: {base}")
    print(f"Target size: {args.target_size_kb} KB")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print("Rule: 1..13700 => No Fire, 13701..53451 => Fire")

    results = []
    for source_dir in (rgb_dir, thermal_dir):
        print(f"\nProcessing: {source_dir}")
        res = process_source_dir(
            source_dir=source_dir,
            target_size_kb=args.target_size_kb,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        results.append(res)

        print(f"  Top-level images found: {res.total_top_level_images}")
        print(f"  Copied: {res.copied}")
        print(f"  Skipped existing: {res.skipped_existing}")
        print(f"  Skipped bad filename: {res.skipped_bad_name}")
        print(f"  Skipped out-of-range index: {res.skipped_out_of_range}")
        print(f"  Failed: {res.failed}")

    total_failed = sum(r.failed for r in results)
    print("\nDone.")
    if total_failed > 0:
        print(f"Completed with failures: {total_failed}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
