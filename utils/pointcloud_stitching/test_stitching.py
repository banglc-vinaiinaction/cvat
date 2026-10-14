# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Unit and Integration Tests for 3D Point Cloud Stitching & Smoothing
Location: utils/pointcloud_stitching/test_stitching.py

Run via:
    python3 utils/pointcloud_stitching/test_stitching.py
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.pointcloud_stitching.pcd_io import read_pcd, write_pcd
from utils.pointcloud_stitching.ego_pose_manager import EgoPoseManager
from utils.pointcloud_stitching.trajectory_smoother import TrajectorySmoother
from utils.pointcloud_stitching.icp_registration import FastICPRegistration
from utils.pointcloud_stitching.cloud_smoother import PointCloudSmoother
from utils.pointcloud_stitching.scene_stitcher import SceneStitcher

class TestPointCloudStitching(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("/tmp/cvat_pcd_test")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def test_pcd_io_ascii_and_binary(self):
        """Kiểm tra đọc và ghi file PCD cả ASCII lẫn Binary."""
        np.random.seed(42)
        pts = np.random.uniform(-10, 10, size=(500, 4)).astype(np.float32)

        # 1. ASCII
        ascii_path = str(self.tmp_dir / "test_ascii.pcd")
        write_pcd(ascii_path, pts, fields=['x', 'y', 'z', 'intensity'], binary=False)
        read_pts_ascii, fields_ascii, _ = read_pcd(ascii_path)

        self.assertEqual(len(read_pts_ascii), 500)
        self.assertEqual(fields_ascii[:4], ['x', 'y', 'z', 'intensity'])
        np.testing.assert_allclose(pts[:, :3], read_pts_ascii[:, :3], atol=1e-3)

        # 2. Binary
        bin_path = str(self.tmp_dir / "test_bin.pcd")
        write_pcd(bin_path, pts, fields=['x', 'y', 'z', 'intensity'], binary=True)
        read_pts_bin, fields_bin, _ = read_pcd(bin_path)

        self.assertEqual(len(read_pts_bin), 500)
        np.testing.assert_allclose(pts[:, :3], read_pts_bin[:, :3], atol=1e-5)

    def test_trajectory_smoother(self):
        """Kiểm tra thuật toán làm mượt quỹ đạo B-Spline và SLERP."""
        smoother = TrajectorySmoother()
        n = 10
        timestamps = [float(i) * 0.1 for i in range(n)]

        # Tạo quỹ đạo thẳng kèm nhiễu rung lắc
        raw_poses = []
        for i, t in enumerate(timestamps):
            T = np.eye(4)
            # Thêm nhiễu ngẫu nhiên vào trục z
            noise_z = np.sin(i * 3.0) * 0.1
            T[:3, 3] = [t * 10.0, 0.0, 1.5 + noise_z]
            raw_poses.append(T)

        smoothed = smoother.smooth_poses(raw_poses, timestamps)
        self.assertEqual(len(smoothed), n)

        # Kiểm tra z sau khi làm mượt có phương sai nhỏ hơn
        raw_z = [p[2, 3] for p in raw_poses]
        smooth_z = [p[2, 3] for p in smoothed]
        self.assertTrue(np.std(np.diff(smooth_z)) <= np.std(np.diff(raw_z)))

    def test_icp_registration(self):
        """Kiểm tra thuật toán vi chỉnh ICP hội tụ chính xác trên mây điểm mẫu."""
        icp = FastICPRegistration(max_iterations=40, max_correspondence_distance=1.0)

        # Tạo mây điểm hình lập phương
        np.random.seed(0)
        target_pts = np.random.uniform(-2, 2, size=(1000, 3)).astype(np.float32)

        # Biến đổi nhân tạo: xoay 5 độ và dịch 0.1m
        theta = np.radians(5.0)
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        t_true = np.array([0.08, -0.05, 0.02])

        source_pts = (target_pts - t_true) @ R_true

        # Chạy ICP từ ma trận Identity
        T_refined, rmse = icp.align(source_pts, target_pts)

        # Kiểm tra RMSE phải rất nhỏ (< 3cm)
        self.assertLess(rmse, 0.05)
        np.testing.assert_allclose(T_refined[:3, 3], t_true, atol=0.03)

    def test_voxel_downsampling_and_sor(self):
        """Kiểm tra Voxel Grid Downsampling và Statistical Outlier Removal."""
        smoother = PointCloudSmoother(voxel_size=0.1, sor_nb_neighbors=10, sor_std_ratio=1.5)

        # Tạo 500 điểm phân bố trong khối [-2, 2]
        np.random.seed(42)
        inlier_pts = np.random.uniform(-2, 2, size=(500, 4)).astype(np.float32)

        # 1. Kiểm tra Voxel Downsample
        downsampled = smoother.voxel_downsample(inlier_pts, voxel_size=0.2)
        self.assertLess(len(downsampled), len(inlier_pts))

        # 2. Thêm 3 điểm nhiễu cô lập cách xa 20m
        outliers = np.array([
            [25.0, 25.0, 25.0, 0.1],
            [-25.0, -25.0, 25.0, 0.1],
            [30.0, -30.0, 15.0, 0.1]
        ], dtype=np.float32)
        test_pts = np.vstack([inlier_pts, outliers])

        # SOR lọc bỏ điểm dị biệt
        clean_pts, mask = smoother.remove_statistical_outliers(test_pts, nb_neighbors=15, std_ratio=1.5)
        # Các điểm cách xa > 15m phải bị loại bỏ hoàn toàn
        max_dist = np.max(np.linalg.norm(clean_pts[:, :3], axis=1))
        self.assertLess(max_dist, 10.0)
        self.assertLess(len(clean_pts), len(test_pts))

    def test_full_stitching_end_to_end(self):
        """Kiểm tra toàn bộ pipeline chạy với dataset mẫu waymo_sample nếu có."""
        poses_file = Path("waymo_sample/extracted/ego_poses.json")
        pcd_dir = Path("waymo_sample/extracted/lidar/pcd")
        if not (poses_file.exists() and pcd_dir.exists()):
            self.skipTest("waymo_sample dataset not found, skipping integration test.")

        mgr = EgoPoseManager(poses_file)
        stitcher = SceneStitcher(
            ego_pose_mgr=mgr,
            window_size=1,
            voxel_size=0.05,
            enable_trajectory_smoothing=True,
            enable_icp=True,
            enable_sor=True
        )

        pcd_map = {}
        for rank, cvat_idx, sample_id, t_sec in mgr.get_chronological_order():
            cand = pcd_dir / f"{sample_id}.pcd"
            if cand.exists():
                pcd_map[cvat_idx] = str(cand)

        # Ghép 1 frame
        first_idx = mgr.get_chronological_order()[0][1]
        stitched_pts, fields, stats = stitcher.stitch_frame(first_idx, pcd_map)

        self.assertGreater(len(stitched_pts), 10000)
        self.assertIn("final_smoothed_points", stats)
        self.assertGreater(stats["raw_accumulated_points"], stats["original_points"])
        print(f"\n[Integration Test Passed] Successfully stitched frame #{first_idx}: {len(stitched_pts)} points.")

    def test_packed_rgb_preservation(self):
        """Kiểm tra bảo toàn màu RGB đóng gói (packed float32/uint32) qua PCD I/O và Voxel Downsampling."""
        smoother = PointCloudSmoother(voxel_size=1.0)
        # Hai điểm cùng nằm trong voxel [0, 0, 0] với màu RGB: (200, 100, 50) và (100, 200, 150)
        c1 = np.array([(200 << 16) | (100 << 8) | 50], dtype=np.uint32).view(np.float32)[0]
        c2 = np.array([(100 << 16) | (200 << 8) | 150], dtype=np.uint32).view(np.float32)[0]
        pts = np.array([
            [0.1, 0.2, 0.3, c1],
            [0.2, 0.3, 0.4, c2],
        ], dtype=np.float32)

        down = smoother.voxel_downsample(pts, voxel_size=1.0, fields=['x', 'y', 'z', 'rgb'])
        self.assertEqual(len(down), 1)
        packed_out = np.ascontiguousarray(down[:, 3], dtype=np.float32).view(np.uint32)[0]
        r_out = (packed_out >> 16) & 0xFF
        g_out = (packed_out >> 8) & 0xFF
        b_out = packed_out & 0xFF
        self.assertEqual((r_out, g_out, b_out), (150, 150, 100))

        # Kiểm tra ghi/đọc ASCII và Binary cho trường 'rgb'
        for mode_bin in (False, True):
            p = str(self.tmp_dir / f"test_rgb_{'bin' if mode_bin else 'ascii'}.pcd")
            write_pcd(p, down, fields=['x', 'y', 'z', 'rgb'], binary=mode_bin)
            loaded, loaded_fields, _ = read_pcd(p)
            self.assertEqual(loaded_fields, ['x', 'y', 'z', 'rgb'])
            u32_loaded = np.ascontiguousarray(loaded[:, 3], dtype=np.float32).view(np.uint32)[0]
            self.assertEqual(
                ((u32_loaded >> 16) & 0xFF, (u32_loaded >> 8) & 0xFF, u32_loaded & 0xFF),
                (150, 150, 100),
            )

if __name__ == "__main__":
    unittest.main()
