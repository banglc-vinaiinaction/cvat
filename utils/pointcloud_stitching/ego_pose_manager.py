# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Ego Pose Manager for CVAT 3D Annotations & Trajectory Tracking
Location: utils/3d_stitching/ego_pose_manager.py

Handles SE(3) coordinate transformations between sensor (LiDAR) coordinates and
global/world coordinates, chronological frame sequencing, and ego-motion compensation.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

class EgoPoseManager:
    def __init__(self, poses_file: Optional[Union[str, Path]] = None):
        if poses_file is None:
            base_dir = Path(__file__).resolve().parent
            candidates = [
                base_dir.parent.parent / "waymo_sample" / "extracted" / "ego_poses.json",
                base_dir.parent.parent / "waymo_sample" / "extracted_80" / "ego_poses.json",
                Path("waymo_sample/extracted/ego_poses.json"),
            ]
            for c in candidates:
                if c.exists():
                    poses_file = c
                    break

        if poses_file is None or not Path(poses_file).exists():
            raise FileNotFoundError(f"Could not locate ego_poses.json. Checked: {candidates}")

        self.poses_path = Path(poses_file).resolve()
        with open(self.poses_path, "r") as f:
            self.data = json.load(f)

        self.frames = self.data.get("frames", [])
        self.frame_by_index: Dict[int, Dict[str, Any]] = {
            f["cvat_frame_index"]: f for f in self.frames
        }
        self.frame_by_sample: Dict[str, Dict[str, Any]] = {
            f.get("sample_id", f"sample_{i}"): f for i, f in enumerate(self.frames)
        }

        # Chronological sequence sorted by true timestamp
        self.chronological_sequence = sorted(
            [
                (
                    f.get("chronological_rank", idx),
                    f["cvat_frame_index"],
                    f.get("sample_id", f"sample_{idx:02d}"),
                    f.get("timestamp_seconds", f.get("timestamp_micros", idx * 0.1 * 1e6) / 1e6)
                )
                for idx, f in enumerate(self.frames)
            ],
            key=lambda x: x[3] # Sort strictly by timestamp
        )

    def get_chronological_order(self) -> List[Tuple[int, int, str, float]]:
        """
        Returns list of frames sorted by true timestamp:
        [(chronological_rank, cvat_frame_index, sample_id, timestamp_seconds), ...]
        """
        return self.chronological_sequence

    def get_world_transform(self, frame_index: int) -> np.ndarray:
        """Returns 4x4 matrix T_{world <- sensor} for given CVAT frame index."""
        frame = self.frame_by_index.get(frame_index)
        if not frame:
            raise KeyError(f"CVAT frame index {frame_index} not found in ego poses.")
        return np.array(frame["pose_transform_world"], dtype=np.float64)

    def get_relative_transform(self, frame_from: int, frame_to: int) -> np.ndarray:
        """
        Returns 4x4 relative transformation T_{to <- from}:
        x_{to} = T_{to <- from} * x_{from}
        """
        T_w_from = self.get_world_transform(frame_from)
        T_w_to = self.get_world_transform(frame_to)
        T_to_w = np.linalg.inv(T_w_to)
        return T_to_w @ T_w_from

    def sensor_to_world_point(self, pt: np.ndarray, frame_index: int) -> np.ndarray:
        """Transforms 3D point [x, y, z] from LiDAR sensor to World coordinates."""
        T = self.get_world_transform(frame_index)
        pt_arr = np.asarray(pt[:3], dtype=np.float64)
        return T[:3, :3] @ pt_arr + T[:3, 3]

    def world_to_sensor_point(self, pt: np.ndarray, frame_index: int) -> np.ndarray:
        """Transforms 3D point [x, y, z] from World to LiDAR sensor coordinates."""
        T = self.get_world_transform(frame_index)
        T_inv = np.linalg.inv(T)
        pt_arr = np.asarray(pt[:3], dtype=np.float64)
        return T_inv[:3, :3] @ pt_arr + T_inv[:3, 3]

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Applies 4x4 transformation matrix to (N, D) array (D >= 3).
        Preserves extra feature columns (intensity, etc.).
        """
        if len(points) == 0:
            return points
        R_mat = np.ascontiguousarray(transform_matrix[:3, :3].T, dtype=np.float32)
        t_vec = transform_matrix[:3, 3].astype(np.float32)
        out = np.empty_like(points, dtype=np.float32)
        pts_xyz = points[:, :3].astype(np.float32, copy=False)
        out[:, :3] = pts_xyz @ R_mat + t_vec
        if points.shape[1] > 3:
            out[:, 3:] = points[:, 3:]
        return out

