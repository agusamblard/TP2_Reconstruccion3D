"""Procesa todos los pares estéreo para generar rectificaciones, disparidades, profundidades y poses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pickle

from pose_utils import (
    PoseEstimationResult,
    create_charuco_board,
    draw_aruco_results,
    draw_reprojection,
    estimate_board_pose,
)
from utils import compute_depth


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def rectify_pair(left_img: np.ndarray, right_img: np.ndarray, maps: Dict[str, np.ndarray]):
    left_rect = cv2.remap(
        left_img,
        maps["left_map_x"],
        maps["left_map_y"],
        interpolation=cv2.INTER_LINEAR,
    )
    right_rect = cv2.remap(
        right_img,
        maps["right_map_x"],
        maps["right_map_y"],
        interpolation=cv2.INTER_LINEAR,
    )
    return left_rect, right_rect


def create_matcher() -> cv2.StereoSGBM:
    window_size = 5
    min_disp = 0
    num_disp = 16 * 12  # Debe ser múltiplo de 16
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=7,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def compute_disparity(left_rect: np.ndarray, right_rect: np.ndarray, matcher: cv2.StereoMatcher) -> np.ndarray:
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def save_visualization(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_npy(data: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)


def disparity_to_color(disparity: np.ndarray) -> np.ndarray:
    disp = disparity.copy()
    disp[disp < 0] = 0
    norm = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)


def depth_to_color(depth: np.ndarray, max_depth: float = 2.0) -> np.ndarray:
    depth_vis = depth.copy()
    depth_vis[~np.isfinite(depth_vis)] = 0
    depth_vis = np.clip(depth_vis, 0, max_depth)
    norm = cv2.normalize(depth_vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def extract_baseline(calibration: Dict[str, np.ndarray]) -> float:
    if "T" in calibration:
        baseline = float(np.linalg.norm(calibration["T"].ravel()))
    elif "translationVector" in calibration:
        baseline = float(np.linalg.norm(calibration["translationVector"].ravel()))
    else:
        raise KeyError("No se encontró la línea base en la calibración")
    return baseline


def compute_all_pairs():
    root = Path("data/stereo_budha_charuco")
    captures = root / "captures"
    rectified_dir = Path("outputs/rectified")
    disparity_dir = Path("outputs/disparity")
    depth_dir = Path("outputs/depth")
    pose_dir = Path("outputs/poses")
    reproj_dir = Path("outputs/reprojection")

    maps = load_pickle(Path("data/pkls/stereo_maps.pkl"))
    calibration = load_pickle(Path("data/pkls/stereo_calibration.pkl"))

    matcher = create_matcher()

    indices = sorted(
        int(p.stem.split("_")[1])
        for p in captures.glob("left_*.jpg")
        if p.stem.startswith("left_")
    )

    rectified_paths: List[Dict[str, str]] = []
    disparity_paths: List[str] = []
    depth_paths: List[str] = []
    pose_results: List[Dict[str, object]] = []

    baseline = extract_baseline(calibration)
    focal_length = float(maps["P1"][0, 0])

    for idx in indices:
        left_path = captures / f"left_{idx}.jpg"
        right_path = captures / f"right_{idx}.jpg"
        if not left_path.exists() or not right_path.exists():
            continue

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        if left_img is None or right_img is None:
            continue

        left_rect, right_rect = rectify_pair(left_img, right_img, maps)

        rect_left_path = rectified_dir / f"rect_left_{idx}.png"
        rect_right_path = rectified_dir / f"rect_right_{idx}.png"
        save_visualization(left_rect, rect_left_path)
        save_visualization(right_rect, rect_right_path)
        rectified_paths.append({
            "index": idx,
            "left": str(rect_left_path),
            "right": str(rect_right_path),
        })

        disparity = compute_disparity(left_rect, right_rect, matcher)
        disp_path = disparity_dir / f"disp_{idx}.npy"
        save_npy(disparity, disp_path)
        disparity_paths.append(str(disp_path))
        disp_color = disparity_to_color(disparity)
        save_visualization(disp_color, disparity_dir / f"disp_{idx}.png")

        depth = compute_depth(disparity, focal_length, baseline, default=np.nan, min_disparity=1e-3)
        depth_path = depth_dir / f"depth_{idx}.npy"
        save_npy(depth, depth_path)
        depth_paths.append(str(depth_path))
        depth_color = depth_to_color(depth)
        save_visualization(depth_color, depth_dir / f"depth_{idx}.png")

    pose_dir.mkdir(parents=True, exist_ok=True)
    reproj_dir.mkdir(parents=True, exist_ok=True)

    if indices:
        selected = np.linspace(0, len(indices) - 1, 5, dtype=int)
        pose_indices = [indices[i] for i in selected]
    else:
        pose_indices = []

    board = create_charuco_board(
        squares_x=5,
        squares_y=7,
        square_length=0.03,
        marker_length=0.02,
    )
    camera_matrix = maps["P1"][:3, :3].astype(np.float32)
    dist_coeffs = np.zeros((1, 5), dtype=np.float32)

    for idx in pose_indices:
        rect_left_path = rectified_dir / f"rect_left_{idx}.png"
        if not rect_left_path.exists():
            continue
        left_img = cv2.imread(str(rect_left_path))
        if left_img is None:
            continue

        pose = estimate_board_pose(left_img, board, camera_matrix, dist_coeffs)
        if pose is None:
            continue

        detection_img = draw_aruco_results(left_img, pose.detection, size=2)
        save_visualization(detection_img, pose_dir / f"pose_detection_{idx}.png")

        reprojection = draw_reprojection(
            left_img,
            board,
            pose.rvec,
            pose.tvec,
            camera_matrix,
            dist_coeffs,
        )
        save_visualization(reprojection, reproj_dir / f"reprojection_{idx}.png")

        pose_results.append(
            {
                "index": idx,
                "method": pose.method,
                "rvec": pose.rvec.reshape(-1).tolist(),
                "tvec": pose.tvec.reshape(-1).tolist(),
            }
        )

    summary = {
        "baseline_m": baseline,
        "focal_px": focal_length,
        "pairs": rectified_paths,
        "disparities": disparity_paths,
        "depths": depth_paths,
        "poses": pose_results,
    }

    with (Path("outputs") / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    compute_all_pairs()
