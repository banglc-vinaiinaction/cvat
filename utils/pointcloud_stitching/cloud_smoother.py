# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Point Cloud Smoothing, Denoising, and Voxel Grid Filtering
Location: utils/3d_stitching/cloud_smoother.py

Cung cấp các thuật toán làm mượt bề mặt và khử nhiễu mây điểm sau khi ghép:
1. Voxel Grid Filter: Chia không gian thành lưới 3D, tính trung bình điểm trong mỗi voxel.
   Giúp làm phẳng mặt đường/tường và khống chế dung lượng file cho CVAT.
2. Statistical Outlier Removal (SOR): Khử các tia bắn xa, bụi và nhiễu cảm biến.
3. Radius Outlier Removal (ROR): Khử các điểm cô lập không đủ mật độ láng giềng.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

class PointCloudSmoother:
    def __init__(
        self,
        voxel_size: float = 0.05,
        sor_nb_neighbors: int = 20,
        sor_std_ratio: float = 2.0
    ):
        """
        Args:
            voxel_size: Kích thước cạnh của mỗi ô voxel (mét), ví dụ 0.05m (5cm)
            sor_nb_neighbors: Số điểm láng giềng xét trong Statistical Outlier Removal
            sor_std_ratio: Hệ số độ lệch chuẩn ngưỡng lọc nhiễu
        """
        self.voxel_size = voxel_size
        self.sor_nb_neighbors = sor_nb_neighbors
        self.std_ratio = sor_std_ratio

    def voxel_downsample(self, points: np.ndarray, voxel_size: Optional[float] = None) -> np.ndarray:
        """
        Làm mượt và giảm mật độ mây điểm bằng Voxel Grid Downsampling.
        Tính centroid (trung bình cộng) của các điểm rơi vào cùng một ô voxel.

        Args:
            points: (N, D) numpy array (D >= 3: x, y, z, ...)
            voxel_size: kích thước voxel (nếu None lấy theo cấu hình mặc định)
        Returns:
            smoothed_points: (M, D) mảng điểm đã được làm phẳng và giảm tải
        """
        if len(points) == 0:
            return points

        vs = voxel_size if voxel_size is not None else self.voxel_size
        if vs <= 0:
            return points

        pts_xyz = points[:, :3]

        # 1. Tính chỉ số voxel nguyên cho từng điểm: [ix, iy, iz]
        voxel_coords = np.floor(pts_xyz / vs).astype(np.int64)

        # 2. Tạo khóa duy nhất 1D int64 cho mỗi voxel (tránh tràn số bằng cách dịch min về 0)
        min_coords = voxel_coords.min(axis=0)
        shifted_coords = voxel_coords - min_coords
        grid_dims = shifted_coords.max(axis=0) + 1

        if float(grid_dims[0]) * float(grid_dims[1]) * float(grid_dims[2]) < float(1 << 62):
            voxel_keys = (
                shifted_coords[:, 0] * (grid_dims[1] * grid_dims[2])
                + shifted_coords[:, 1] * grid_dims[2]
                + shifted_coords[:, 2]
            )
            _, inverse_indices, counts = np.unique(
                voxel_keys, return_inverse=True, return_counts=True
            )
        else:
            contig = np.ascontiguousarray(shifted_coords)
            void_view = contig.view(np.dtype((np.void, contig.dtype.itemsize * 3))).ravel()
            _, inverse_indices, counts = np.unique(
                void_view, return_inverse=True, return_counts=True
            )

        num_voxels = len(counts)

        # 3. Tính trung bình các trường (x, y, z, intensity, ...) trong từng voxel bằng np.bincount
        num_features = points.shape[1]
        pts_f64 = points.astype(np.float64, copy=False)
        inv_counts = (1.0 / counts)[:, np.newaxis]

        sum_features = np.column_stack([
            np.bincount(inverse_indices, weights=pts_f64[:, col], minlength=num_voxels)
            for col in range(num_features)
        ])

        smoothed_points = (sum_features * inv_counts).astype(np.float32)
        return smoothed_points

    def remove_statistical_outliers(
        self,
        points: np.ndarray,
        nb_neighbors: Optional[int] = None,
        std_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lọc nhiễu thống kê SOR (Statistical Outlier Removal).
        Loại bỏ các điểm có khoảng cách trung bình tới k láng giềng vượt quá (mean + ratio * std).

        Args:
            points: (N, D) numpy array
            nb_neighbors: số láng giềng k (mặc định 20)
            std_ratio: ngưỡng độ lệch chuẩn (mặc định 2.0)
        Returns:
            filtered_points: (M, D) mảng điểm đã lọc sạch nhiễu
            mask: boolean array (N,) đánh dấu các điểm giữ lại
        """
        if len(points) <= 10:
            return points, np.ones(len(points), dtype=bool)

        k = nb_neighbors if nb_neighbors is not None else self.sor_nb_neighbors
        ratio = std_ratio if std_ratio is not None else self.std_ratio
        k = min(k, len(points) - 1)

        pts_xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        tree = cKDTree(pts_xyz, leafsize=48, compact_nodes=True, balanced_tree=False)
        # Tìm k+1 láng giềng (bao gồm chính nó ở vị trí đầu tiên)
        distances, _ = tree.query(pts_xyz, k=k + 1, workers=-1)

        # Bỏ cột 0 (khoảng cách tới chính nó = 0)
        neighbor_distances = distances[:, 1:]
        mean_distances = np.mean(neighbor_distances, axis=1)

        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)

        threshold = global_mean + ratio * global_std
        inlier_mask = mean_distances <= threshold

        return points[inlier_mask], inlier_mask

    def remove_radius_outliers(
        self,
        points: np.ndarray,
        radius: float = 0.5,
        min_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lọc nhiễu bán kính ROR (Radius Outlier Removal).
        Loại bỏ các điểm có ít hơn min_neighbors láng giềng trong bán kính radius.
        """
        if len(points) == 0:
            return points, np.zeros(0, dtype=bool)

        pts_xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        tree = cKDTree(pts_xyz, leafsize=48, compact_nodes=True, balanced_tree=False)
        counts = tree.query_ball_point(pts_xyz, r=radius, return_length=True, workers=-1)
        # Trừ chính điểm đó (1)
        inlier_mask = (counts - 1) >= min_neighbors
        return points[inlier_mask], inlier_mask

