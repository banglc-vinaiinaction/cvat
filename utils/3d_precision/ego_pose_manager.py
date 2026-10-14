#!/usr/bin/env python3
"""
Ego Pose Manager for CVAT 3D Annotations & Trajectory Tracking
Location: cvat-annotator-dashboard-3d/ego_pose_manager.py

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
            # Try default locations
            base_dir = Path(__file__).resolve().parent
            candidates = [
                base_dir / "data" / "ego_poses.json",
                base_dir.parent / "waymo_sample" / "extracted" / "ego_poses.json",
            ]
            for c in candidates:
                if c.exists():
                    poses_file = c
                    break

        if poses_file is None or not Path(poses_file).exists():
            raise FileNotFoundError(f"Could not locate ego_poses.json in default paths.")

        self.poses_path = Path(poses_file)
        with open(self.poses_path, "r") as f:
            self.data = json.load(f)

        self.frames = self.data.get("frames", [])
        self.frame_by_index: Dict[int, Dict[str, Any]] = {f["cvat_frame_index"]: f for f in self.frames}
        self.frame_by_sample: Dict[str, Dict[str, Any]] = {f["sample_id"]: f for f in self.frames}
        
        # Chronological order list of (chronological_rank, cvat_frame_index, sample_id, timestamp_seconds)
        self.chronological_sequence = sorted(
            [(f["chronological_rank"], f["cvat_frame_index"], f["sample_id"], f["timestamp_seconds"]) for f in self.frames],
            key=lambda x: x[0]
        )

    def get_chronological_order(self) -> List[Tuple[int, int, str, float]]:
        """
        Returns list of frames sorted by true time:
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
        """
        Transforms 3D point [x, y, z] from LiDAR sensor coordinates to World coordinates.
        """
        T = self.get_world_transform(frame_index)
        homo = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
        world_homo = T @ homo
        return world_homo[:3]

    def world_to_sensor_point(self, pt: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Transforms 3D point [x, y, z] from World coordinates to LiDAR sensor coordinates.
        """
        T = self.get_world_transform(frame_index)
        T_inv = np.linalg.inv(T)
        homo = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
        sensor_homo = T_inv @ homo
        return sensor_homo[:3]

    def sensor_to_world_box(self, center: np.ndarray, yaw: float, frame_index: int) -> Tuple[np.ndarray, float]:
        """
        Transforms bounding box center [x, y, z] and yaw heading to World coordinates.
        """
        T = self.get_world_transform(frame_index)
        # Transform center
        homo = np.array([center[0], center[1], center[2], 1.0], dtype=np.float64)
        world_center = (T @ homo)[:3]
        
        # Transform yaw
        ego_yaw = self.frame_by_index[frame_index]["yaw_rad"]
        world_yaw = (yaw + ego_yaw + np.pi) % (2 * np.pi) - np.pi
        return world_center, world_yaw

    def world_to_sensor_box(self, center: np.ndarray, yaw: float, frame_index: int) -> Tuple[np.ndarray, float]:
        """
        Transforms bounding box center [x, y, z] and yaw heading from World to Sensor coordinates.
        """
        T = self.get_world_transform(frame_index)
        T_inv = np.linalg.inv(T)
        homo = np.array([center[0], center[1], center[2], 1.0], dtype=np.float64)
        sensor_center = (T_inv @ homo)[:3]
        
        ego_yaw = self.frame_by_index[frame_index]["yaw_rad"]
        sensor_yaw = (yaw - ego_yaw + np.pi) % (2 * np.pi) - np.pi
        return sensor_center, sensor_yaw

    def print_summary(self):
        print(f"=== EgoPoseManager ({len(self.frames)} frames loaded) ===")
        print(f"Source: {self.poses_path}")
        print("\nCVAT Frame -> Chronological Timeline:")
        for rank, cvat_idx, sample_id, t_sec in self.chronological_sequence:
            f = self.frame_by_index[cvat_idx]
            pos = f["translation_world"]
            vel = f["speed_mps"]
            print(f"  Time Rank {rank}: CVAT Frame #{cvat_idx:02d} ({sample_id}) | t={t_sec:.3f}s | Pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | Speed={vel:.2f} m/s")

if __name__ == "__main__":
    mgr = EgoPoseManager()
    mgr.print_summary()
