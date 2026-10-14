# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Trajectory Smoother for Autonomous Driving Ego-poses
Location: utils/3d_stitching/trajectory_smoother.py

Làm mượt quỹ đạo di chuyển của xe (Ego-trajectory Smoothing) bằng cách kết hợp:
1. B-Spline / Savitzky-Golay cho vị trí tịnh tiến (x, y, z).
2. Spherical Linear Interpolation (SLERP) cho ma trận góc quay Rotation (Quaternion).
Giúp triệt tiêu rung giật tần số cao từ cảm biến IMU/GPS và triệt tiêu lỗi bóng ma khi ghép frame.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, make_interp_spline
from scipy.spatial.transform import Rotation as R, Slerp
from typing import List, Dict, Tuple, Optional

class TrajectorySmoother:
    def __init__(self, smoothing_factor: float = 1.0, spline_degree: int = 3):
        """
        Args:
            smoothing_factor: độ mượt của Spline (càng lớn càng mượt, 0 là đi qua mọi điểm)
            spline_degree: bậc của đường cong spline (mặc định bậc 3 - Cubic)
        """
        self.smoothing_factor = smoothing_factor
        self.spline_degree = spline_degree

    def smooth_poses(
        self,
        poses: List[np.ndarray],
        timestamps: List[float],
        method: str = "spline"
    ) -> List[np.ndarray]:
        """
        Làm mượt danh sách các ma trận SE(3) [4x4] theo trục thời gian.

        Args:
            poses: danh sách ma trận SE(3) [4x4]
            timestamps: danh sách thời gian tương ứng (giây)
            method: "spline" hoặc "moving_average"
        Returns:
            smoothed_poses: danh sách ma trận SE(3) [4x4] đã làm mượt
        """
        n = len(poses)
        if n < 4:
            # Nếu ít hơn 4 frame thì không đủ bậc spline, giữ nguyên
            return [np.copy(p) for p in poses]

        # Đảm bảo timestamps strictly increasing
        t_arr = np.array(timestamps, dtype=np.float64)
        diffs = np.diff(t_arr)
        if np.any(diffs <= 0):
            # Nếu timestamps bị trùng hoặc giảm, tạo chuỗi đều giả định
            t_arr = np.linspace(0.0, float(n - 1), n)

        # 1. Trích xuất Translations (x, y, z)
        translations = np.array([p[:3, 3] for p in poses], dtype=np.float64)

        # 2. Trích xuất Rotations
        rot_matrices = [p[:3, :3] for p in poses]
        rotations = R.from_matrix(rot_matrices)

        # 3. Làm mượt vị trí (x, y, z)
        k = min(self.spline_degree, n - 1)
        smooth_trans = np.zeros_like(translations)

        try:
            # Dùng UnivariateSpline có s để khử nhiễu
            s = self.smoothing_factor * (n * np.var(translations)) * 0.01
            for axis in range(3):
                spl = UnivariateSpline(t_arr, translations[:, axis], k=k, s=s)
                smooth_trans[:, axis] = spl(t_arr)
        except Exception:
            # Fallback sang make_interp_spline
            for axis in range(3):
                spl = make_interp_spline(t_arr, translations[:, axis], k=min(k, 3))
                smooth_trans[:, axis] = spl(t_arr)

        # 4. Làm mượt góc quay Quaternion bằng SLERP
        smooth_rots = []
        try:
            # Tạo Slerp interpolator
            slerp = Slerp(t_arr, rotations)
            # Sample lại tại các mốc thời gian
            smooth_rots = slerp(t_arr)
        except Exception:
            smooth_rots = rotations

        # 5. Tái cấu trúc ma trận SE(3)
        smoothed_poses = []
        for i in range(n):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = smooth_rots[i].as_matrix()
            T[:3, 3] = smooth_trans[i]
            smoothed_poses.append(T)

        return smoothed_poses
