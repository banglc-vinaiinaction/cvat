# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Fast Rigid ICP (Iterative Closest Point) Registration
Location: utils/3d_stitching/icp_registration.py

Thuật toán vi chỉnh căn khớp mây điểm (Fine-registration) dùng cKDTree và Kabsch SVD.
Giúp triệt tiêu hoàn toàn sai số trôi (drift) giữa 2 frame kề nhau, ngăn chặn hiện tượng
cột điện / bờ tường bị "nhân đôi" (Double walls/Ghosting) sau khi stitch.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

class FastICPRegistration:
    def __init__(
        self,
        max_iterations: int = 30,
        max_correspondence_distance: float = 0.4,
        tolerance: float = 1e-4,
        subsample_size: int = 15000
    ):
        """
        Args:
            max_iterations: Số vòng lặp tối đa
            max_correspondence_distance: Khoảng cách tối đa (mét) để coi 2 điểm là tương ứng
            tolerance: Ngưỡng dừng khi độ thay đổi ma trận biến đổi nhỏ hơn giá trị này
            subsample_size: Lấy mẫu ngẫu nhiên số điểm để tăng tốc độ tính toán KDTree
        """
        self.max_iterations = max_iterations
        self.max_correspondence_distance = max_correspondence_distance
        self.tolerance = tolerance
        self.subsample_size = subsample_size

    def _kabsch_svd(self, p_src: np.ndarray, p_tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thuật toán Kabsch tìm nghiệm đóng tối ưu cho R, t giữa 2 tập điểm tương ứng:
        min || R * p_src + t - p_tgt ||^2
        """
        centroid_src = np.mean(p_src, axis=0)
        centroid_tgt = np.mean(p_tgt, axis=0)

        src_centered = p_src - centroid_src
        tgt_centered = p_tgt - centroid_tgt

        # Ma trận hiệp phương sai H
        H = src_centered.T @ tgt_centered

        # Phân tích suy biến SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Xử lý trường hợp ma trận phản xạ (det(R) == -1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_tgt - R @ centroid_src
        return R, t

    def align(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        init_transform: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Căn khớp source_points vào target_points.

        Args:
            source_points: (N, 3) tọa độ điểm nguồn
            target_points: (M, 3) tọa độ điểm đích
            init_transform: (4, 4) ma trận biến đổi thô ban đầu (từ Egopose)
        Returns:
            refined_transform: (4, 4) ma trận SE(3) đã được vi chỉnh
            rmse: sai số Root-Mean-Square Error cuối cùng
        """
        src_xyz = source_points[:, :3].astype(np.float64)
        tgt_xyz = target_points[:, :3].astype(np.float64)

        # Lấy mẫu ngẫu nhiên nếu số điểm quá lớn để đạt hiệu năng cao
        if len(src_xyz) > self.subsample_size:
            idx = np.random.choice(len(src_xyz), self.subsample_size, replace=False)
            src_sample = src_xyz[idx]
        else:
            src_sample = src_xyz

        if len(tgt_xyz) > self.subsample_size * 2:
            idx_tgt = np.random.choice(len(tgt_xyz), self.subsample_size * 2, replace=False)
            tgt_sample = tgt_xyz[idx_tgt]
        else:
            tgt_sample = tgt_xyz

        # Xây dựng KD-Tree trên điểm target
        target_kdtree = cKDTree(tgt_sample)

        # Khởi tạo ma trận biến đổi tích lũy
        current_transform = np.copy(init_transform) if init_transform is not None else np.eye(4, dtype=np.float64)

        prev_rmse = float('inf')
        rmse = float('inf')

        for it in range(self.max_iterations):
            # Biến đổi source theo ma trận hiện tại
            homo = np.hstack([src_sample, np.ones((len(src_sample), 1))])
            transformed_src = (current_transform @ homo.T).T[:, :3]

            # Tìm điểm gần nhất trong target KD-Tree
            distances, indices = target_kdtree.query(transformed_src, k=1, workers=-1)

            # Lọc các cặp điểm nằm trong bán kính hợp lệ
            valid_mask = distances < self.max_correspondence_distance
            num_valid = np.sum(valid_mask)

            if num_valid < 50:
                # Không đủ cặp điểm tương ứng, dừng và dùng ma trận hiện tại
                break

            matched_src = transformed_src[valid_mask]
            matched_tgt = tgt_sample[indices[valid_mask]]

            rmse = float(np.sqrt(np.mean(distances[valid_mask] ** 2)))

            # Tính delta R, t
            delta_R, delta_t = self._kabsch_svd(matched_src, matched_tgt)

            delta_T = np.eye(4, dtype=np.float64)
            delta_T[:3, :3] = delta_R
            delta_T[:3, 3] = delta_t

            # Cập nhật: current_transform = delta_T * current_transform
            current_transform = delta_T @ current_transform

            # Kiểm tra hội tụ
            if abs(prev_rmse - rmse) < self.tolerance:
                break
            prev_rmse = rmse

        return current_transform, rmse
