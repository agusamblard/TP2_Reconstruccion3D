"""Utilidades para detectar tableros ChArUco y estimar su pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class PoseEstimationResult:
    """Resultado de estimar la pose de un tablero."""

    detection: Dict[str, np.ndarray]
    rvec: np.ndarray
    tvec: np.ndarray
    method: str


def _board_dictionary(board, default=cv2.aruco.DICT_6X6_250):
    """Devuelve el diccionario ArUco asociado al tablero."""

    try:
        return board.dictionary
    except AttributeError:
        try:
            return board.getDictionary()
        except AttributeError:
            return cv2.aruco.getPredefinedDictionary(default)


def _board_ids(board) -> np.ndarray:
    """Ids de marcadores del tablero."""

    try:
        return np.array(board.ids).ravel()
    except AttributeError:
        return np.array(board.getIds()).ravel()


def _board_objpoints(board):
    """Coordenadas 3D (metros) de cada marcador del tablero."""

    try:
        obj_pts = board.objPoints
    except AttributeError:
        obj_pts = board.getObjPoints()
    return [pts.reshape(-1, 3) for pts in obj_pts]


def _board_charuco_corners(board) -> np.ndarray:
    """Devuelve los vértices de las casillas del tablero en 3D."""

    try:
        return board.chessboardCorners
    except AttributeError:
        return board.getChessboardCorners()


def create_charuco_board(
    squares_x: int = 5,
    squares_y: int = 7,
    square_length: float = 0.04,
    marker_length: float = 0.02,
    dictionary_type: int = cv2.aruco.DICT_6X6_250,
):
    """Crea un tablero ChArUco usando unidades en metros."""

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, aruco_dict
    )


def detect_charuco_markers(image: np.ndarray, board) -> Optional[Dict[str, np.ndarray]]:
    """Detecta marcadores ArUco pertenecientes al tablero."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    try:
        detector_params = cv2.aruco.DetectorParameters()
    except Exception:
        detector_params = cv2.aruco.DetectorParameters_create()

    aruco_dict = _board_dictionary(board)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

    if ids is None or len(ids) == 0:
        return None

    return {"corners": corners, "ids": ids, "rejected": rejected}


def draw_aruco_results(
    image: np.ndarray,
    detection: Optional[Dict[str, np.ndarray]],
    size: int = 5,
) -> np.ndarray:
    """Dibuja la detección de marcadores sobre la imagen original."""

    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    if detection is None:
        return result

    corners = detection["corners"]
    ids = detection["ids"]

    if corners is not None and len(corners) > 0:
        ids = ids.flatten()
        for marker_corners, marker_id in zip(corners, ids):
            pts = marker_corners.reshape((4, 2))
            tl, tr, br, bl = pts
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            cv2.line(result, tl, tr, (0, 0, 255), size)
            cv2.line(result, tr, br, (0, 255, 0), size)
            cv2.line(result, br, bl, (255, 0, 0), size)
            cv2.line(result, bl, tl, (0, 255, 255), size)
            cx = int((tl[0] + br[0]) / 2.0)
            cy = int((tl[1] + br[1]) / 2.0)
            cv2.circle(result, (cx, cy), 3 * size, (0, 0, 255), -1)
            org = (cx + 15, max(cy + 30, 0))
            cv2.putText(
                result,
                str(marker_id),
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (0, 0, 0),
                size + 10,
            )
            cv2.putText(
                result,
                str(marker_id),
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (0, 255, 0),
                size + 3,
            )
    return result


def estimate_pose_homography(
    board,
    detection: Dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    undistort: bool = False,
) -> Optional[PoseEstimationResult]:
    """Estima la pose del tablero usando homografía entre marcadores."""

    corners_list = detection["corners"]
    ids = detection["ids"]

    if ids is None or len(ids) < 4:
        return None

    board_ids = _board_ids(board)
    board_obj_points = _board_objpoints(board)
    id_to_index = {int(k): i for i, k in enumerate(board_ids)}

    image_points, board_points = [], []

    for marker_corners, marker_id in zip(corners_list, ids.ravel()):
        marker_id = int(marker_id)
        if marker_id not in id_to_index:
            continue
        obj_marker_3d = board_obj_points[id_to_index[marker_id]]
        img_marker_2d = marker_corners.reshape(-1, 2)
        board_points.append(obj_marker_3d[:, :2])
        image_points.append(img_marker_2d)

    if len(image_points) == 0:
        return None

    image_points = np.concatenate(image_points, axis=0).astype(np.float32)
    board_points = np.concatenate(board_points, axis=0).astype(np.float32)

    if image_points.shape[0] < 4:
        return None

    if undistort:
        image_points = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
        ).reshape(-1, 2)

    homography, _ = cv2.findHomography(board_points, image_points, method=cv2.LMEDS)
    if homography is None:
        return None

    Hn = np.linalg.inv(camera_matrix) @ homography
    h1, h2, h3 = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    lam = 1.0 / max(np.linalg.norm(h1), 1e-9)
    r1, r2 = h1 * lam, h2 * lam
    r3 = np.cross(r1, r2)
    r_approx = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(r_approx)
    rotation = u @ vt
    translation = h3 * lam

    rvec, _ = cv2.Rodrigues(rotation.astype(np.float64))
    tvec = translation.reshape(3, 1).astype(np.float64)

    return PoseEstimationResult(detection=detection, rvec=rvec, tvec=tvec, method="homography")


def estimate_board_pose(
    image: np.ndarray,
    board,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[PoseEstimationResult]:
    """Intenta estimar la pose del tablero ChArUco."""

    detection = detect_charuco_markers(image, board)
    if detection is None:
        return None

    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        interp = cv2.aruco.interpolateCornersCharuco(
            markerCorners=detection["corners"],
            markerIds=detection["ids"],
            image=gray,
            board=board,
        )
        if interp is not None:
            count, charuco_corners, charuco_ids = interp
            if (
                charuco_corners is not None
                and charuco_ids is not None
                and count is not None
                and count >= 4
            ):
                rvec_init = np.zeros((3, 1), dtype=np.float64)
                tvec_init = np.zeros((3, 1), dtype=np.float64)
                ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners.astype(np.float32),
                    charuco_ids.astype(np.int32).reshape(-1, 1),
                    board,
                    camera_matrix,
                    dist_coeffs,
                    rvec_init,
                    tvec_init,
                )
                if ok:
                    return PoseEstimationResult(
                        detection=detection, rvec=rvec, tvec=tvec, method="charuco"
                    )

    return estimate_pose_homography(board, detection, camera_matrix, dist_coeffs)


def draw_reprojection(
    image: np.ndarray,
    board,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float = 0.05,
) -> np.ndarray:
    """Superpone la proyección del tablero y los ejes sobre la imagen."""

    result = image.copy()

    corners3d = _board_charuco_corners(board)
    projected, _ = cv2.projectPoints(corners3d, rvec, tvec, camera_matrix, dist_coeffs)
    for pt in projected.reshape(-1, 2):
        cv2.circle(result, tuple(np.round(pt).astype(int)), 4, (0, 255, 0), -1)

    axis = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    axis_img, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(np.round(axis_img[0, 0]).astype(int))
    x_axis = tuple(np.round(axis_img[1, 0]).astype(int))
    y_axis = tuple(np.round(axis_img[2, 0]).astype(int))
    z_axis = tuple(np.round(axis_img[3, 0]).astype(int))
    cv2.line(result, origin, x_axis, (0, 0, 255), 3)
    cv2.line(result, origin, y_axis, (0, 255, 0), 3)
    cv2.line(result, origin, z_axis, (255, 0, 0), 3)

    return result