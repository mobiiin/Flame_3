#!/usr/bin/env python3
"""
Align paired RGB and thermal images into a shared thermal-centric frame.

Assumptions and caveats:
- This method assumes RGB and thermal sensors are rigidly mounted, or at least
  consistently paired with limited viewpoint changes.
- Homography is most valid when parallax is limited and scene depth variation
  is not extreme. Large depth changes can break global projective alignment.
- Severe smoke, low thermal texture, and weak cross-modal structure reduce
  feature match reliability and can cause registration failure.
- For highest geometric accuracy, full intrinsic/extrinsic calibration and
  cross-modal calibration are preferred when available.

This script is designed for cross-modal registration where direct RGB/thermal
intensity correspondence is weak. It uses modality-robust preprocessing
(contrast normalization + gradients + edges), feature matching with RANSAC
transform fitting, optional ECC refinement, and explicit quality controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_RGB_DIR = "/home/mhabibp/Downloads/Flame 2/Original Sized RGB Images"
DEFAULT_THERMAL_DIR = "/home/mhabibp/Downloads/Flame 2/Original Sized Thermal Images"
DEFAULT_OUTPUT_DIR = "./aligned_rgb_thermal_output"


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    rel_key: str
    rgb_path: Path
    thermal_path: Path


@dataclass
class MatchResult:
    detector_name: str
    rgb_keypoints: int
    thermal_keypoints: int
    raw_matches: int
    good_matches: int
    src_pts: np.ndarray
    dst_pts: np.ndarray


@dataclass
class TransformResult:
    success: bool
    transform: Optional[np.ndarray]
    transform_type: Optional[str]
    inlier_mask: Optional[np.ndarray]
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch align RGB/thermal image pairs with robust multimodal registration."
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path(DEFAULT_RGB_DIR),
        help=f"Path to RGB image root folder (default: {DEFAULT_RGB_DIR}).",
    )
    parser.add_argument(
        "--thermal-dir",
        type=Path,
        default=Path(DEFAULT_THERMAL_DIR),
        help=f"Path to thermal image root folder (default: {DEFAULT_THERMAL_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--pairing-mode",
        choices=["filename", "sorted"],
        default="filename",
        help="Pairing strategy: filename stem matching or sorted order pairing.",
    )
    parser.add_argument(
        "--transform-preference",
        choices=["homography", "affine"],
        default="homography",
        help="Preferred transform model. Script will fallback if preferred model is unstable.",
    )
    parser.add_argument(
        "--resize-final-to-thermal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, save aligned outputs at thermal resolution. "
            "If false, outputs are resampled to RGB size while keeping thermal footprint."
        ),
    )
    parser.add_argument(
        "--attempt-ecc-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ECC refinement from the feature-based initialization.",
    )
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save RGB/thermal overlay previews for QA.",
    )
    parser.add_argument(
        "--save-valid-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save valid overlap masks for warped RGB.",
    )
    parser.add_argument(
        "--save-thermal-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save normalized thermal preview images for quick visual QA.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan input folders for images.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum number of pairs to process (0 = all).",
    )
    parser.add_argument(
        "--min-good-matches",
        type=int,
        default=12,
        help="Minimum good matches required before transform estimation.",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=10,
        help="Minimum inliers required for a successful registration.",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.20,
        help="Minimum inlier ratio (inliers / good_matches).",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=6.0,
        help="Maximum mean reprojection error (pixels) to accept registration.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.995,
        help="RANSAC confidence for transform estimation.",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.5,
        help="CLAHE clip limit used in multimodal preprocessing.",
    )
    parser.add_argument(
        "--clahe-grid-size",
        type=int,
        default=8,
        help="CLAHE tile grid size (single integer for square grid).",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=50,
        help="Canny lower threshold.",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
        help="Canny upper threshold.",
    )
    parser.add_argument(
        "--ecc-iterations",
        type=int,
        default=120,
        help="Maximum ECC iterations.",
    )
    parser.add_argument(
        "--ecc-eps",
        type=float,
        default=1e-6,
        help="ECC convergence epsilon.",
    )
    parser.add_argument(
        "--ecc-min-corr-gain",
        type=float,
        default=1e-4,
        help="Minimum alignment-correlation gain required to keep ECC result.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path, level_name: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rgb_thermal_align")
    logger.setLevel(getattr(logging, level_name.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "alignment.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def collect_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if is_image_file(p)]
    else:
        files = [p for p in root.iterdir() if is_image_file(p)]
    return sorted(files, key=lambda p: p.as_posix().lower())


def relative_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.with_suffix("").as_posix().lower()


def basename_stem(path: Path) -> str:
    return path.stem.lower()


def discover_pairs(
    rgb_dir: Path,
    thermal_dir: Path,
    pairing_mode: str,
    recursive: bool,
    logger: logging.Logger,
) -> List[ImagePair]:
    rgb_files = collect_images(rgb_dir, recursive=recursive)
    thermal_files = collect_images(thermal_dir, recursive=recursive)
    logger.info("Found %d RGB files and %d thermal files.", len(rgb_files), len(thermal_files))

    if not rgb_files or not thermal_files:
        return []

    pairs: List[ImagePair] = []
    if pairing_mode == "sorted":
        n = min(len(rgb_files), len(thermal_files))
        if len(rgb_files) != len(thermal_files):
            logger.warning(
                "Sorted pairing: count mismatch (RGB=%d, Thermal=%d). Using first %d pairs.",
                len(rgb_files),
                len(thermal_files),
                n,
            )
        for i in range(n):
            rel_key = f"pair_{i + 1:06d}"
            pairs.append(
                ImagePair(
                    pair_id=rel_key,
                    rel_key=rel_key,
                    rgb_path=rgb_files[i],
                    thermal_path=thermal_files[i],
                )
            )
        return pairs

    rgb_rel_map: Dict[str, Path] = {}
    thermal_rel_map: Dict[str, Path] = {}
    rgb_rel_dupes: set[str] = set()
    thermal_rel_dupes: set[str] = set()

    for p in rgb_files:
        key = relative_stem(p, rgb_dir)
        if key in rgb_rel_map:
            rgb_rel_dupes.add(key)
        else:
            rgb_rel_map[key] = p

    for p in thermal_files:
        key = relative_stem(p, thermal_dir)
        if key in thermal_rel_map:
            thermal_rel_dupes.add(key)
        else:
            thermal_rel_map[key] = p

    common_rel = sorted(set(rgb_rel_map.keys()) & set(thermal_rel_map.keys()))
    common_rel = [k for k in common_rel if k not in rgb_rel_dupes and k not in thermal_rel_dupes]

    if common_rel:
        for key in common_rel:
            pairs.append(
                ImagePair(
                    pair_id=key,
                    rel_key=key,
                    rgb_path=rgb_rel_map[key],
                    thermal_path=thermal_rel_map[key],
                )
            )
        logger.info("Paired %d images using relative filename stems.", len(pairs))
        return pairs

    logger.warning(
        "No relative-stem matches found. Falling back to unique basename stem matching."
    )

    rgb_stem_map: Dict[str, Path] = {}
    thermal_stem_map: Dict[str, Path] = {}
    rgb_stem_dupes: set[str] = set()
    thermal_stem_dupes: set[str] = set()

    for p in rgb_files:
        key = basename_stem(p)
        if key in rgb_stem_map:
            rgb_stem_dupes.add(key)
        else:
            rgb_stem_map[key] = p

    for p in thermal_files:
        key = basename_stem(p)
        if key in thermal_stem_map:
            thermal_stem_dupes.add(key)
        else:
            thermal_stem_map[key] = p

    common_stems = sorted(set(rgb_stem_map.keys()) & set(thermal_stem_map.keys()))
    common_stems = [k for k in common_stems if k not in rgb_stem_dupes and k not in thermal_stem_dupes]
    for key in common_stems:
        pairs.append(
            ImagePair(
                pair_id=key,
                rel_key=key,
                rgb_path=rgb_stem_map[key],
                thermal_path=thermal_stem_map[key],
            )
        )

    if rgb_stem_dupes or thermal_stem_dupes:
        logger.warning(
            "Duplicate basename stems detected and skipped (RGB dupes=%d, Thermal dupes=%d).",
            len(rgb_stem_dupes),
            len(thermal_stem_dupes),
        )
    logger.info("Paired %d images using basename stems.", len(pairs))
    return pairs


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v <= min_v + 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - min_v) / (max_v - min_v)
    arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def convert_thermal_to_gray_for_registration(thermal: np.ndarray) -> np.ndarray:
    if thermal.ndim == 2:
        return thermal
    if thermal.ndim == 3 and thermal.shape[2] == 1:
        return thermal[:, :, 0]
    if thermal.ndim == 3 and thermal.shape[2] == 3:
        # Handles false-color thermal JPEGs by reducing to luminance-like grayscale.
        return cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    if thermal.ndim == 3 and thermal.shape[2] == 4:
        return cv2.cvtColor(thermal, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported thermal image shape: {thermal.shape}")


def gradient_magnitude_u8(gray_u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def preprocess_for_multimodal_registration(
    rgb_bgr: np.ndarray,
    thermal_raw: np.ndarray,
    clahe_clip_limit: float,
    clahe_grid_size: int,
    canny_low: int,
    canny_high: int,
) -> Dict[str, np.ndarray]:
    """
    Build modality-robust representations.

    Why this helps:
    - RGB/thermal intensities are not directly comparable.
    - CLAHE improves local contrast in both modalities.
    - Gradient/edge representations emphasize structural geometry that transfers
      better across modalities than raw intensities.
    """
    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    thermal_gray = convert_thermal_to_gray_for_registration(thermal_raw)

    rgb_u8 = normalize_to_uint8(rgb_gray)
    thermal_u8 = normalize_to_uint8(thermal_gray)

    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clahe_clip_limit)),
        tileGridSize=(max(2, clahe_grid_size), max(2, clahe_grid_size)),
    )
    rgb_eq = clahe.apply(rgb_u8)
    thermal_eq = clahe.apply(thermal_u8)

    rgb_grad = gradient_magnitude_u8(rgb_eq)
    thermal_grad = gradient_magnitude_u8(thermal_eq)

    rgb_edges = cv2.Canny(rgb_eq, threshold1=canny_low, threshold2=canny_high, L2gradient=True)
    thermal_edges = cv2.Canny(
        thermal_eq, threshold1=canny_low, threshold2=canny_high, L2gradient=True
    )

    rgb_repr = cv2.addWeighted(rgb_grad, 0.72, rgb_edges, 0.28, 0.0)
    thermal_repr = cv2.addWeighted(thermal_grad, 0.72, thermal_edges, 0.28, 0.0)

    rgb_repr = cv2.GaussianBlur(rgb_repr, (3, 3), 0.0)
    thermal_repr = cv2.GaussianBlur(thermal_repr, (3, 3), 0.0)

    return {
        "rgb_gray_u8": rgb_u8,
        "thermal_gray_u8": thermal_u8,
        "rgb_repr": rgb_repr,
        "thermal_repr": thermal_repr,
    }


def create_detector() -> Tuple[str, Any, int, float]:
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=12)
        return "SIFT", detector, cv2.NORM_L2, 0.75
    detector = cv2.ORB_create(
        nfeatures=10000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=10,
    )
    return "ORB", detector, cv2.NORM_HAMMING, 0.85


def detect_and_match_features(rgb_repr: np.ndarray, thermal_repr: np.ndarray) -> MatchResult:
    detector_name, detector, norm_type, ratio_thresh = create_detector()
    kp_rgb, des_rgb = detector.detectAndCompute(rgb_repr, None)
    kp_thermal, des_thermal = detector.detectAndCompute(thermal_repr, None)

    kp_rgb = kp_rgb or []
    kp_thermal = kp_thermal or []
    if des_rgb is None or des_thermal is None or len(kp_rgb) < 4 or len(kp_thermal) < 4:
        return MatchResult(
            detector_name=detector_name,
            rgb_keypoints=len(kp_rgb),
            thermal_keypoints=len(kp_thermal),
            raw_matches=0,
            good_matches=0,
            src_pts=np.empty((0, 2), dtype=np.float32),
            dst_pts=np.empty((0, 2), dtype=np.float32),
        )

    matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    knn_matches = matcher.knnMatch(des_rgb, des_thermal, k=2)
    raw_count = len(knn_matches)

    good: List[cv2.DMatch] = []
    for m_n in knn_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    good.sort(key=lambda m: m.distance)
    src_pts = np.float32([kp_rgb[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp_thermal[m.trainIdx].pt for m in good]).reshape(-1, 2)

    return MatchResult(
        detector_name=detector_name,
        rgb_keypoints=len(kp_rgb),
        thermal_keypoints=len(kp_thermal),
        raw_matches=raw_count,
        good_matches=len(good),
        src_pts=src_pts,
        dst_pts=dst_pts,
    )


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float,
    confidence: float,
) -> TransformResult:
    if src_pts.shape[0] < 4:
        return TransformResult(False, None, None, None, "Not enough points for homography.")
    h, inliers = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=10000,
        confidence=confidence,
    )
    if h is None or inliers is None:
        return TransformResult(False, None, None, None, "findHomography failed.")
    if not np.all(np.isfinite(h)):
        return TransformResult(False, None, None, None, "Homography contains non-finite values.")

    h = h.astype(np.float64)
    if abs(h[2, 2]) < 1e-10:
        return TransformResult(False, None, None, None, "Unstable homography scale term.")

    h /= h[2, 2]
    cond = np.linalg.cond(h)
    if not np.isfinite(cond) or cond > 1e10:
        return TransformResult(False, None, None, None, f"Homography ill-conditioned (cond={cond:.2e}).")

    return TransformResult(True, h, "homography", inliers.ravel().astype(bool), "ok")


def estimate_affine(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float,
    confidence: float,
) -> TransformResult:
    if src_pts.shape[0] < 3:
        return TransformResult(False, None, None, None, "Not enough points for affine.")

    a, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=10000,
        confidence=confidence,
        refineIters=15,
    )
    if a is None or inliers is None:
        a, inliers = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=10000,
            confidence=confidence,
            refineIters=15,
        )
    if a is None or inliers is None:
        return TransformResult(False, None, None, None, "estimateAffine2D failed.")
    if not np.all(np.isfinite(a)):
        return TransformResult(False, None, None, None, "Affine contains non-finite values.")

    t = np.eye(3, dtype=np.float64)
    t[:2, :] = a.astype(np.float64)
    return TransformResult(True, t, "affine", inliers.ravel().astype(bool), "ok")


def validate_transform_geometry(
    transform: np.ndarray,
    transform_type: str,
    rgb_shape: Tuple[int, int],
    thermal_shape: Tuple[int, int],
) -> Tuple[bool, str]:
    h_rgb, w_rgb = rgb_shape
    h_th, w_th = thermal_shape

    corners = np.float32(
        [
            [0, 0],
            [w_rgb - 1, 0],
            [w_rgb - 1, h_rgb - 1],
            [0, h_rgb - 1],
        ]
    ).reshape(-1, 1, 2)

    if transform_type == "homography":
        warped = cv2.perspectiveTransform(corners, transform.astype(np.float64)).reshape(-1, 2)
    else:
        warped = cv2.transform(corners, transform[:2, :].astype(np.float64)).reshape(-1, 2)

    if not np.all(np.isfinite(warped)):
        return False, "Warped corners are non-finite."

    area = abs(cv2.contourArea(warped.astype(np.float32)))
    thermal_area = float(max(1, w_th * h_th))
    area_ratio = area / thermal_area
    if area_ratio < 0.02 or area_ratio > 8.0:
        return False, f"Unreasonable projected area ratio ({area_ratio:.4f})."

    min_xy = np.min(warped, axis=0)
    max_xy = np.max(warped, axis=0)
    span = max_xy - min_xy
    if span[0] > 15 * w_th or span[1] > 15 * h_th:
        return False, "Projected bounds are implausibly large."

    return True, "ok"


def estimate_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    preference: str,
    ransac_threshold: float,
    confidence: float,
    rgb_shape: Tuple[int, int],
    thermal_shape: Tuple[int, int],
) -> TransformResult:
    if preference == "homography":
        first = estimate_homography(src_pts, dst_pts, ransac_threshold, confidence)
        if first.success and first.transform is not None and first.transform_type is not None:
            ok, reason = validate_transform_geometry(
                first.transform, first.transform_type, rgb_shape, thermal_shape
            )
            if ok:
                return first
        second = estimate_affine(src_pts, dst_pts, ransac_threshold, confidence)
        if second.success and second.transform is not None and second.transform_type is not None:
            ok, reason = validate_transform_geometry(
                second.transform, second.transform_type, rgb_shape, thermal_shape
            )
            if ok:
                return second
            return TransformResult(False, None, None, None, f"Affine fallback invalid: {reason}")
        return TransformResult(False, None, None, None, f"Homography failed; affine fallback failed: {second.reason}")

    first = estimate_affine(src_pts, dst_pts, ransac_threshold, confidence)
    if first.success and first.transform is not None and first.transform_type is not None:
        ok, reason = validate_transform_geometry(
            first.transform, first.transform_type, rgb_shape, thermal_shape
        )
        if ok:
            return first

    second = estimate_homography(src_pts, dst_pts, ransac_threshold, confidence)
    if second.success and second.transform is not None and second.transform_type is not None:
        ok, reason = validate_transform_geometry(
            second.transform, second.transform_type, rgb_shape, thermal_shape
        )
        if ok:
            return second
        return TransformResult(False, None, None, None, f"Homography fallback invalid: {reason}")
    return TransformResult(False, None, None, None, f"Affine failed; homography fallback failed: {second.reason}")


def apply_transform_points(points_xy: np.ndarray, transform: np.ndarray, transform_type: str) -> np.ndarray:
    pts = points_xy.reshape(-1, 1, 2).astype(np.float64)
    if transform_type == "homography":
        out = cv2.perspectiveTransform(pts, transform.astype(np.float64))
    else:
        out = cv2.transform(pts, transform[:2, :].astype(np.float64))
    return out.reshape(-1, 2)


def compute_reprojection_error(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    inlier_mask: np.ndarray,
    transform: np.ndarray,
    transform_type: str,
) -> Tuple[float, float]:
    if src_pts.shape[0] == 0 or inlier_mask is None:
        return float("inf"), float("inf")
    inlier_mask = inlier_mask.astype(bool)
    if np.count_nonzero(inlier_mask) == 0:
        return float("inf"), float("inf")

    src_i = src_pts[inlier_mask]
    dst_i = dst_pts[inlier_mask]
    proj = apply_transform_points(src_i, transform, transform_type)
    err = np.linalg.norm(proj - dst_i, axis=1)
    return float(np.mean(err)), float(np.median(err))


def warp_image(
    image: np.ndarray,
    transform: np.ndarray,
    transform_type: str,
    out_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    w, h = out_size
    if transform_type == "homography":
        return cv2.warpPerspective(
            image,
            transform.astype(np.float64),
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return cv2.warpAffine(
        image,
        transform[:2, :].astype(np.float64),
        (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_alignment_correlation(
    rgb_repr: np.ndarray,
    thermal_repr: np.ndarray,
    transform: np.ndarray,
    transform_type: str,
) -> float:
    h, w = thermal_repr.shape[:2]
    warped = warp_image(rgb_repr, transform, transform_type, out_size=(w, h), interpolation=cv2.INTER_LINEAR)
    mask_in = np.full(rgb_repr.shape[:2], 255, dtype=np.uint8)
    valid_mask = warp_image(
        mask_in,
        transform,
        transform_type,
        out_size=(w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    valid = valid_mask > 0
    if np.count_nonzero(valid) < 100:
        return -1.0

    a = warped[valid].astype(np.float32)
    b = thermal_repr[valid].astype(np.float32)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-6 or sb < 1e-6:
        return -1.0
    corr = float(np.mean((a / sa) * (b / sb)))
    return corr


def refine_alignment_with_ecc(
    rgb_repr: np.ndarray,
    thermal_repr: np.ndarray,
    transform: np.ndarray,
    transform_type: str,
    ecc_iterations: int,
    ecc_eps: float,
    min_corr_gain: float,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    metrics: Dict[str, Any] = {
        "ecc_attempted": True,
        "ecc_accepted": False,
        "ecc_reason": "",
        "ecc_initial_corr": None,
        "ecc_final_corr": None,
    }

    initial_corr = compute_alignment_correlation(rgb_repr, thermal_repr, transform, transform_type)
    metrics["ecc_initial_corr"] = initial_corr

    rgb_f = rgb_repr.astype(np.float32) / 255.0
    thermal_f = thermal_repr.astype(np.float32) / 255.0
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(10, int(ecc_iterations)),
        max(1e-9, float(ecc_eps)),
    )

    try:
        if transform_type == "homography":
            motion = cv2.MOTION_HOMOGRAPHY
            warp_init = transform.astype(np.float32).copy()
        else:
            motion = cv2.MOTION_AFFINE
            warp_init = transform[:2, :].astype(np.float32).copy()

        cc, warp_out = cv2.findTransformECC(
            templateImage=thermal_f,
            inputImage=rgb_f,
            warpMatrix=warp_init,
            motionType=motion,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )

        if transform_type == "homography":
            refined = warp_out.astype(np.float64)
        else:
            refined = np.eye(3, dtype=np.float64)
            refined[:2, :] = warp_out.astype(np.float64)
        final_corr = compute_alignment_correlation(rgb_repr, thermal_repr, refined, transform_type)
        metrics["ecc_final_corr"] = final_corr

        if not np.isfinite(final_corr):
            metrics["ecc_reason"] = "ECC produced non-finite correlation."
            return transform, False, metrics

        if final_corr >= initial_corr + float(min_corr_gain):
            metrics["ecc_accepted"] = True
            metrics["ecc_reason"] = f"Accepted (gain={final_corr - initial_corr:.6f}, ECC={cc:.6f})."
            return refined, True, metrics

        metrics["ecc_reason"] = (
            f"Rejected (corr gain {final_corr - initial_corr:.6f} < {min_corr_gain:.6f})."
        )
        return transform, False, metrics

    except cv2.error as exc:
        metrics["ecc_reason"] = f"ECC failed: {exc}"
        return transform, False, metrics


def build_overlay(aligned_rgb: np.ndarray, thermal_gray_u8: np.ndarray) -> np.ndarray:
    thermal_color = cv2.applyColorMap(thermal_gray_u8, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(aligned_rgb, 0.60, thermal_color, 0.40, 0.0)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def output_paths(output_dir: Path, rel_key: str) -> Dict[str, Path]:
    rel_path = Path(rel_key).with_suffix(".png")
    rel_json = Path(rel_key).with_suffix(".json")
    return {
        "aligned_rgb": output_dir / "aligned_rgb" / rel_path,
        "thermal_reference": output_dir / "thermal_reference" / rel_path,
        "thermal_preview": output_dir / "thermal_preview" / rel_path,
        "overlay": output_dir / "overlay_preview" / rel_path,
        "valid_mask": output_dir / "valid_mask" / rel_path,
        "metrics": output_dir / "metrics" / rel_json,
    }


def make_output_canvas_transform(
    transform_rgb_to_thermal: np.ndarray,
    rgb_size: Tuple[int, int],
    thermal_size: Tuple[int, int],
    resize_final_to_thermal: bool,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    w_rgb, h_rgb = rgb_size
    w_th, h_th = thermal_size
    if resize_final_to_thermal:
        return transform_rgb_to_thermal, (w_th, h_th)

    sx = w_rgb / max(1.0, float(w_th))
    sy = h_rgb / max(1.0, float(h_th))
    scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    scaled_transform = scale @ transform_rgb_to_thermal
    return scaled_transform, (w_rgb, h_rgb)


def process_single_pair(
    pair: ImagePair,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    started = time.perf_counter()
    metrics: Dict[str, Any] = {
        "pair_id": pair.pair_id,
        "rel_key": pair.rel_key,
        "rgb_path": str(pair.rgb_path),
        "thermal_path": str(pair.thermal_path),
        "status": "failed",
        "failure_reason": "",
        "transform_preference": args.transform_preference,
        "transform_type_used": None,
        "detector": None,
        "rgb_keypoints": 0,
        "thermal_keypoints": 0,
        "raw_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reprojection_error_mean": float("inf"),
        "reprojection_error_median": float("inf"),
        "ecc_attempted": bool(args.attempt_ecc_refinement),
        "ecc_accepted": False,
        "ecc_initial_corr": None,
        "ecc_final_corr": None,
        "ecc_reason": "",
        "output_width": 0,
        "output_height": 0,
        "processing_time_ms": 0.0,
    }

    try:
        rgb = cv2.imread(str(pair.rgb_path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(pair.thermal_path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            metrics["failure_reason"] = "Failed to read RGB image."
            return metrics
        if thermal is None:
            metrics["failure_reason"] = "Failed to read thermal image."
            return metrics

        prep = preprocess_for_multimodal_registration(
            rgb_bgr=rgb,
            thermal_raw=thermal,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_grid_size=args.clahe_grid_size,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )

        match = detect_and_match_features(prep["rgb_repr"], prep["thermal_repr"])
        metrics["detector"] = match.detector_name
        metrics["rgb_keypoints"] = match.rgb_keypoints
        metrics["thermal_keypoints"] = match.thermal_keypoints
        metrics["raw_matches"] = match.raw_matches
        metrics["good_matches"] = match.good_matches

        if match.good_matches < max(4, args.min_good_matches):
            metrics["failure_reason"] = (
                f"Insufficient good matches ({match.good_matches} < {args.min_good_matches})."
            )
            return metrics

        rgb_h, rgb_w = rgb.shape[:2]
        thermal_h, thermal_w = prep["thermal_gray_u8"].shape[:2]

        tr = estimate_transform(
            src_pts=match.src_pts,
            dst_pts=match.dst_pts,
            preference=args.transform_preference,
            ransac_threshold=args.ransac_threshold,
            confidence=args.ransac_confidence,
            rgb_shape=(rgb_h, rgb_w),
            thermal_shape=(thermal_h, thermal_w),
        )
        if not tr.success or tr.transform is None or tr.inlier_mask is None or tr.transform_type is None:
            metrics["failure_reason"] = f"Transform estimation failed: {tr.reason}"
            return metrics

        transform = tr.transform
        transform_type = tr.transform_type
        inlier_mask = tr.inlier_mask.astype(bool)
        inliers = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inliers / max(1, match.good_matches)
        rep_mean, rep_median = compute_reprojection_error(
            src_pts=match.src_pts,
            dst_pts=match.dst_pts,
            inlier_mask=inlier_mask,
            transform=transform,
            transform_type=transform_type,
        )

        metrics["transform_type_used"] = transform_type
        metrics["inliers"] = inliers
        metrics["inlier_ratio"] = float(inlier_ratio)
        metrics["reprojection_error_mean"] = float(rep_mean)
        metrics["reprojection_error_median"] = float(rep_median)

        if inliers < args.min_inliers:
            metrics["failure_reason"] = f"Insufficient inliers ({inliers} < {args.min_inliers})."
            return metrics
        if inlier_ratio < args.min_inlier_ratio:
            metrics["failure_reason"] = (
                f"Inlier ratio too low ({inlier_ratio:.3f} < {args.min_inlier_ratio:.3f})."
            )
            return metrics
        if rep_mean > args.max_reprojection_error:
            metrics["failure_reason"] = (
                f"Reprojection error too high ({rep_mean:.3f} > {args.max_reprojection_error:.3f})."
            )
            return metrics

        if args.attempt_ecc_refinement:
            refined, accepted, ecc_m = refine_alignment_with_ecc(
                rgb_repr=prep["rgb_repr"],
                thermal_repr=prep["thermal_repr"],
                transform=transform,
                transform_type=transform_type,
                ecc_iterations=args.ecc_iterations,
                ecc_eps=args.ecc_eps,
                min_corr_gain=args.ecc_min_corr_gain,
            )
            metrics["ecc_accepted"] = bool(accepted)
            metrics["ecc_initial_corr"] = ecc_m.get("ecc_initial_corr")
            metrics["ecc_final_corr"] = ecc_m.get("ecc_final_corr")
            metrics["ecc_reason"] = ecc_m.get("ecc_reason", "")
            if accepted:
                ok, reason = validate_transform_geometry(
                    refined, transform_type, (rgb_h, rgb_w), (thermal_h, thermal_w)
                )
                if ok:
                    transform = refined
                else:
                    metrics["ecc_accepted"] = False
                    metrics["ecc_reason"] = f"ECC rejected due to geometry: {reason}"

        transform_out, out_size = make_output_canvas_transform(
            transform_rgb_to_thermal=transform,
            rgb_size=(rgb_w, rgb_h),
            thermal_size=(thermal_w, thermal_h),
            resize_final_to_thermal=args.resize_final_to_thermal,
        )
        out_w, out_h = out_size

        aligned_rgb = warp_image(
            image=rgb,
            transform=transform_out,
            transform_type=transform_type,
            out_size=(out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
        valid_mask = warp_image(
            image=np.full((rgb_h, rgb_w), 255, dtype=np.uint8),
            transform=transform_out,
            transform_type=transform_type,
            out_size=(out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )
        valid_mask = (valid_mask > 0).astype(np.uint8) * 255

        if args.resize_final_to_thermal:
            thermal_for_pair = thermal.copy()
            thermal_gray_for_overlay = prep["thermal_gray_u8"]
        else:
            thermal_for_pair = cv2.resize(
                thermal,
                dsize=(out_w, out_h),
                interpolation=cv2.INTER_LINEAR,
            )
            thermal_gray_for_overlay = cv2.resize(
                prep["thermal_gray_u8"],
                dsize=(out_w, out_h),
                interpolation=cv2.INTER_LINEAR,
            )

        thermal_preview = cv2.applyColorMap(thermal_gray_for_overlay, cv2.COLORMAP_INFERNO)
        overlay = build_overlay(aligned_rgb, thermal_gray_for_overlay)

        metrics["status"] = "success"
        metrics["failure_reason"] = ""
        metrics["aligned_rgb"] = aligned_rgb
        metrics["thermal_for_pair"] = thermal_for_pair
        metrics["thermal_preview"] = thermal_preview
        metrics["overlay"] = overlay
        metrics["valid_mask_img"] = valid_mask
        metrics["output_width"] = int(out_w)
        metrics["output_height"] = int(out_h)

    except Exception as exc:
        metrics["failure_reason"] = f"Unhandled exception: {exc}"
        logger.exception("Error processing pair '%s': %s", pair.pair_id, exc)
    finally:
        metrics["processing_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

    return metrics


def save_outputs(
    output_dir: Path,
    pair: ImagePair,
    metrics: Dict[str, Any],
    save_overlays: bool,
    save_valid_mask: bool,
    save_thermal_preview: bool,
) -> Dict[str, str]:
    out = output_paths(output_dir, pair.rel_key)
    saved_paths: Dict[str, str] = {}

    metrics_for_json = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}

    if metrics["status"] == "success":
        ensure_parent(out["aligned_rgb"])
        ensure_parent(out["thermal_reference"])
        ok1 = cv2.imwrite(str(out["aligned_rgb"]), metrics["aligned_rgb"])
        ok2 = cv2.imwrite(str(out["thermal_reference"]), metrics["thermal_for_pair"])
        if not ok1 or not ok2:
            raise RuntimeError("Failed writing aligned RGB or thermal reference image.")
        saved_paths["aligned_rgb"] = str(out["aligned_rgb"])
        saved_paths["thermal_reference"] = str(out["thermal_reference"])

        if save_thermal_preview:
            ensure_parent(out["thermal_preview"])
            if cv2.imwrite(str(out["thermal_preview"]), metrics["thermal_preview"]):
                saved_paths["thermal_preview"] = str(out["thermal_preview"])

        if save_overlays:
            ensure_parent(out["overlay"])
            if cv2.imwrite(str(out["overlay"]), metrics["overlay"]):
                saved_paths["overlay_preview"] = str(out["overlay"])

        if save_valid_mask:
            ensure_parent(out["valid_mask"])
            if cv2.imwrite(str(out["valid_mask"]), metrics["valid_mask_img"]):
                saved_paths["valid_mask"] = str(out["valid_mask"])

    write_json(out["metrics"], metrics_for_json)
    saved_paths["metrics"] = str(out["metrics"])
    return saved_paths


def process_dataset(args: argparse.Namespace, logger: logging.Logger) -> None:
    rgb_dir = args.rgb_dir.resolve()
    thermal_dir = args.thermal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    if not thermal_dir.is_dir():
        raise FileNotFoundError(f"Thermal directory does not exist: {thermal_dir}")

    pairs = discover_pairs(
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        pairing_mode=args.pairing_mode,
        recursive=args.recursive,
        logger=logger,
    )
    if not pairs:
        raise RuntimeError("No valid RGB/thermal pairs found.")

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
        logger.info("Limiting processing to first %d pairs.", len(pairs))

    logger.info("Processing %d pairs.", len(pairs))

    records: List[Dict[str, Any]] = []
    iterator: Iterable[ImagePair]
    if tqdm is not None:
        iterator = tqdm(pairs, desc="Aligning", unit="pair")
    else:
        iterator = pairs

    for i, pair in enumerate(iterator, start=1):
        result = process_single_pair(pair=pair, args=args, logger=logger)
        try:
            save_map = save_outputs(
                output_dir=output_dir,
                pair=pair,
                metrics=result,
                save_overlays=args.save_overlays,
                save_valid_mask=args.save_valid_mask,
                save_thermal_preview=args.save_thermal_preview,
            )
            result["saved_paths"] = save_map
        except Exception as exc:
            logger.exception("Failed writing outputs for pair '%s': %s", pair.pair_id, exc)
            result["saved_paths"] = {}
            if result.get("status") == "success":
                result["status"] = "failed"
                result["failure_reason"] = f"Output write failure: {exc}"
        records.append(result)

        if result["status"] == "success":
            logger.info(
                "[%d/%d] SUCCESS %s | model=%s inliers=%d ratio=%.3f reproj=%.3f",
                i,
                len(pairs),
                pair.pair_id,
                result["transform_type_used"],
                result["inliers"],
                result["inlier_ratio"],
                result["reprojection_error_mean"],
            )
        else:
            logger.warning(
                "[%d/%d] FAILED  %s | reason=%s",
                i,
                len(pairs),
                pair.pair_id,
                result["failure_reason"],
            )

    success_count = sum(1 for r in records if r["status"] == "success")
    fail_count = len(records) - success_count
    success_rate = success_count / max(1, len(records))

    csv_path = output_dir / "summary_metrics.csv"
    json_path = output_dir / "summary_report.json"

    csv_fields = [
        "pair_id",
        "rel_key",
        "status",
        "failure_reason",
        "rgb_path",
        "thermal_path",
        "detector",
        "transform_preference",
        "transform_type_used",
        "rgb_keypoints",
        "thermal_keypoints",
        "raw_matches",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "reprojection_error_mean",
        "reprojection_error_median",
        "ecc_attempted",
        "ecc_accepted",
        "ecc_initial_corr",
        "ecc_final_corr",
        "ecc_reason",
        "output_width",
        "output_height",
        "processing_time_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in csv_fields}
            writer.writerow(row)

    json_args = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    aggregate = {
        "generated_at_unix": int(time.time()),
        "arguments": json_args,
        "totals": {
            "pairs_processed": len(records),
            "success_count": success_count,
            "failure_count": fail_count,
            "success_rate": success_rate,
        },
        "paths": {
            "rgb_dir": str(rgb_dir),
            "thermal_dir": str(thermal_dir),
            "output_dir": str(output_dir),
            "summary_csv": str(csv_path),
            "summary_json": str(json_path),
            "log_file": str(output_dir / "logs" / "alignment.log"),
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    logger.info("Completed. Success=%d, Failed=%d, Success rate=%.2f%%", success_count, fail_count, 100 * success_rate)
    logger.info("Summary CSV: %s", csv_path)
    logger.info("Summary JSON: %s", json_path)


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.output_dir, args.log_level)
    logger.info("Starting RGB/Thermal alignment pipeline.")
    logger.info("RGB root: %s", args.rgb_dir.resolve())
    logger.info("Thermal root: %s", args.thermal_dir.resolve())
    logger.info("Output root: %s", args.output_dir.resolve())
    process_dataset(args, logger)


if __name__ == "__main__":
    main()
