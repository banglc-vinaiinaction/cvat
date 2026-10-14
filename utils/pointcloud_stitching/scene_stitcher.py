# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Complete Scene Stitcher Pipeline with Trajectory & Cloud Smoothing
Location: utils/3d_stitching/scene_stitcher.py

Pipeline tích hợp toàn diện:
1. Đọc Ego-poses & đồng bộ hóa dòng thời gian (Chronological Sequencing).
2. Làm mượt quỹ đạo xe (B-Spline & Quaternion SLERP).
3. Vi chỉnh căn khớp đa khung hình bằng ICP Registration (chống bóng ma).
4. Ghép nối mây điểm (Scene Stitching) trong cửa sổ trượt (sliding window).
5. Làm mượt bề mặt bằng Voxel Grid Centroid Downsampling.
6. Lọc nhiễu thống kê SOR.
7. Xuất file PCD chuẩn cho CVAT 3D Canvas.
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable, Any

from .pcd_io import read_pcd, write_pcd
from .ego_pose_manager import EgoPoseManager
from .trajectory_smoother import TrajectorySmoother
from .icp_registration import FastICPRegistration
from .cloud_smoother import PointCloudSmoother

class SceneStitcher:
    def __init__(
        self,
        ego_pose_mgr: EgoPoseManager,
        window_size: int = 2,
        voxel_size: float = 0.05,
        enable_trajectory_smoothing: bool = True,
        enable_icp: bool = True,
        enable_sor: bool = True,
        sor_nb_neighbors: int = 20,
        sor_std_ratio: float = 2.0,
        icp_max_distance: float = 0.4
    ):
        """
        Args:
            ego_pose_mgr: Instance của EgoPoseManager đã nạp poses
            window_size: Bán kính khung hình lân cận cần ghép (ví dụ 2 là ghép t-2, t-1, t, t+1, t+2)
            voxel_size: Kích thước voxel để làm phẳng mặt đường và giảm tải (mét)
            enable_trajectory_smoothing: Bật làm mượt B-spline & SLERP trên pose
            enable_icp: Bật vi chỉnh ICP giữa các frame kề nhau
            enable_sor: Bật lọc nhiễu Statistical Outlier Removal
        """
        self.ego_pose_mgr = ego_pose_mgr
        self.window_size = window_size
        self.voxel_size = voxel_size
        self.enable_trajectory_smoothing = enable_trajectory_smoothing
        self.enable_icp = enable_icp
        self.enable_sor = enable_sor

        self.traj_smoother = TrajectorySmoother()
        self.icp = FastICPRegistration(max_correspondence_distance=icp_max_distance)
        self.cloud_smoother = PointCloudSmoother(
            voxel_size=voxel_size,
            sor_nb_neighbors=sor_nb_neighbors,
            sor_std_ratio=sor_std_ratio
        )

        # Chuẩn bị danh sách pose theo đúng thứ tự thời gian
        self.chrono_sequence = self.ego_pose_mgr.get_chronological_order()
        self._init_poses()

    def _init_poses(self):
        """Khởi tạo và làm mượt quỹ đạo pose nếu được kích hoạt."""
        self.raw_poses = []
        self.timestamps = []
        self.frame_indices = []

        for rank, cvat_idx, sample_id, t_sec in self.chrono_sequence:
            self.frame_indices.append(cvat_idx)
            self.timestamps.append(t_sec)
            self.raw_poses.append(self.ego_pose_mgr.get_world_transform(cvat_idx))

        if self.enable_trajectory_smoothing:
            self.smoothed_poses = self.traj_smoother.smooth_poses(self.raw_poses, self.timestamps)
        else:
            self.smoothed_poses = [np.copy(p) for p in self.raw_poses]

        # Tạo map tra cứu pose đã làm mượt theo cvat_frame_index
        self.pose_by_index: Dict[int, np.ndarray] = {
            cvat_idx: self.smoothed_poses[i] for i, cvat_idx in enumerate(self.frame_indices)
        }

    def get_relative_transform(self, frame_from: int, frame_to: int) -> np.ndarray:
        """Tính ma trận biến đổi T_{to <- from} sử dụng pose đã làm mượt."""
        T_w_from = self.pose_by_index[frame_from]
        T_w_to = self.pose_by_index[frame_to]
        T_to_w = np.linalg.inv(T_w_to)
        return T_to_w @ T_w_from

    def stitch_frame(
        self,
        target_cvat_index: int,
        pcd_files_map: Dict[int, str],
        on_status: Optional[Callable[[str], None]] = None
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Ghép mây điểm của target_frame với các frame lân cận.

        Args:
            target_cvat_index: frame index của khung hình mục tiêu
            pcd_files_map: dict map cvat_frame_index -> đường dẫn file .pcd
            on_status: callback cập nhật tiến độ
        Returns:
            stitched_points: mảng điểm sau khi ghép và làm mượt
            fields: tên các trường (x, y, z, intensity, ...)
            stats: thông số thống kê (số điểm trước/sau, rmse, thời gian)
        """
        start_time = time.time()
        target_pcd_path = pcd_files_map.get(target_cvat_index)
        if not target_pcd_path or not os.path.exists(target_pcd_path):
            raise FileNotFoundError(f"PCD file for target frame {target_cvat_index} not found.")

        target_points, fields, meta = read_pcd(target_pcd_path)
        original_count = len(target_points)

        # Xác định vị trí của target_frame trong chuỗi thời gian
        try:
            target_pos = self.frame_indices.index(target_cvat_index)
        except ValueError:
            target_pos = 0

        # Lấy các frame lân cận theo cửa sổ thời gian [-window_size, +window_size]
        start_pos = max(0, target_pos - self.window_size)
        end_pos = min(len(self.frame_indices), target_pos + self.window_size + 1)
        neighbor_indices = [
            self.frame_indices[i] for i in range(start_pos, end_pos)
            if self.frame_indices[i] != target_cvat_index and self.frame_indices[i] in pcd_files_map
        ]

        accumulated_list = [target_points]
        icp_errors = []

        for n_idx in neighbor_indices:
            n_pcd_path = pcd_files_map[n_idx]
            if not os.path.exists(n_pcd_path):
                continue

            n_points, _, _ = read_pcd(n_pcd_path)
            if len(n_points) == 0:
                continue

            # Ma trận biến đổi thô từ pose đã làm mượt: T_{target <- neighbor}
            T_rel = self.get_relative_transform(frame_from=n_idx, frame_to=target_cvat_index)

            # Vi chỉnh bằng ICP nếu được kích hoạt
            if self.enable_icp:
                refined_T, rmse = self.icp.align(
                    source_points=n_points,
                    target_points=target_points,
                    init_transform=T_rel
                )
                icp_errors.append(rmse)
                T_apply = refined_T
            else:
                T_apply = T_rel

            # Biến đổi tọa độ các điểm sang hệ tọa độ của target frame
            transformed_n_points = self.ego_pose_mgr.transform_points(n_points, T_apply)
            accumulated_list.append(transformed_n_points)

        # Ghép tất cả các điểm lại
        merged_points = np.vstack(accumulated_list)
        pre_filter_count = len(merged_points)

        # Thuật toán làm mượt bề mặt: Voxel Grid Downsampling
        if self.voxel_size > 0:
            smoothed_points = self.cloud_smoother.voxel_downsample(merged_points, self.voxel_size)
        else:
            smoothed_points = merged_points

        # Thuật toán lọc nhiễu: Statistical Outlier Removal
        if self.enable_sor:
            final_points, _ = self.cloud_smoother.remove_statistical_outliers(smoothed_points)
        else:
            final_points = smoothed_points

        elapsed = time.time() - start_time
        stats = {
            "target_frame": target_cvat_index,
            "neighbors_used": len(neighbor_indices),
            "original_points": original_count,
            "raw_accumulated_points": pre_filter_count,
            "final_smoothed_points": len(final_points),
            "mean_icp_rmse": float(np.mean(icp_errors)) if icp_errors else 0.0,
            "elapsed_seconds": elapsed,
        }

        if on_status:
            on_status(f"Frame #{target_cvat_index:02d}: {original_count} -> {len(final_points)} pts ({elapsed:.2f}s)")

        return final_points, fields, stats

    def process_sequence(
        self,
        pcd_files_map: Dict[int, str],
        output_dir: Union[str, Path],
        binary: bool = False,
        on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chạy stitching và làm mượt cho toàn bộ danh sách frame.

        Args:
            pcd_files_map: dict map frame index sang file pcd gốc
            output_dir: thư mục lưu trữ các file pcd đã ghép
            binary: lưu định dạng binary để tối ưu tốc độ và dung lượng
            on_progress: callback cập nhật tiến độ (current, total, stats)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(self.frame_indices)

        for i, cvat_idx in enumerate(self.frame_indices):
            if cvat_idx not in pcd_files_map:
                continue

            stitched_pts, fields, stats = self.stitch_frame(cvat_idx, pcd_files_map)
            out_filename = output_path / f"frame_{cvat_idx:06d}.pcd"
            write_pcd(str(out_filename), stitched_pts, fields=fields, binary=binary)
            stats["output_file"] = str(out_filename)
            results.append(stats)

            if on_progress:
                on_progress(i + 1, total, stats)

        return results
