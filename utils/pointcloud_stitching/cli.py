#!/usr/bin/env python3
# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Command-line Interface for 3D Point Cloud Scene Stitching and Smoothing
Location: utils/3d_stitching/cli.py

Example usage:
  python3 utils/3d_stitching/cli.py \
      --poses waymo_sample/extracted/ego_poses.json \
      --pcd-dir waymo_sample/extracted/lidar/pcd \
      --output-dir waymo_sample/extracted/lidar/stitched_pcd \
      --window 2 \
      --voxel-size 0.05 \
      --enable-icp \
      --enable-sor
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.pointcloud_stitching.ego_pose_manager import EgoPoseManager
from utils.pointcloud_stitching.scene_stitcher import SceneStitcher

def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud Scene Stitching with Trajectory & Surface Smoothing for CVAT"
    )
    parser.add_argument(
        "--poses",
        type=str,
        default="waymo_sample/extracted/ego_poses.json",
        help="Path to ego_poses.json file"
    )
    parser.add_argument(
        "--pcd-dir",
        type=str,
        default="waymo_sample/extracted/lidar/pcd",
        help="Directory containing original input PCD files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="waymo_sample/extracted/lidar/stitched_pcd",
        help="Directory to save stitched and smoothed PCD files"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Sliding window radius (e.g. 2 means +/- 2 neighbor frames are stitched, default: 2)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel downsampling size in meters (default: 0.05m = 5cm, 0 to disable)"
    )
    parser.add_argument(
        "--no-traj-smooth",
        action="store_true",
        help="Disable B-Spline and SLERP trajectory smoothing"
    )
    parser.add_argument(
        "--no-icp",
        action="store_true",
        help="Disable ICP registration fine-tuning"
    )
    parser.add_argument(
        "--no-sor",
        action="store_true",
        help="Disable Statistical Outlier Removal filter"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Save output PCDs in binary format for smaller size and faster loading"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to process (useful for quick testing)"
    )

    args = parser.parse_args()

    poses_path = Path(args.poses)
    if not poses_path.exists():
        print(f"[Error] Poses file not found: {poses_path}")
        sys.exit(1)

    pcd_dir = Path(args.pcd_dir)
    if not pcd_dir.exists():
        print(f"[Error] PCD directory not found: {pcd_dir}")
        sys.exit(1)

    print("==================================================================")
    print("  CVAT 3D Scene Stitcher & Multi-level Smoothing Pipeline")
    print("==================================================================")
    print(f"Poses file:         {poses_path}")
    print(f"PCD input dir:      {pcd_dir}")
    print(f"PCD output dir:     {args.output_dir}")
    print(f"Window size:        +/- {args.window} frames")
    print(f"Voxel size:         {args.voxel_size} m")
    print(f"Traj smoothing:     {not args.no_traj_smooth}")
    print(f"ICP fine-tuning:    {not args.no_icp}")
    print(f"SOR outlier filter: {not args.no_sor}")
    print(f"Output format:      {'Binary' if args.binary else 'ASCII'}")
    print("------------------------------------------------------------------")

    # 1. Nạp EgoPoseManager
    print("[1/3] Loading Ego Pose Manager...")
    mgr = EgoPoseManager(poses_path)
    seq = mgr.get_chronological_order()
    print(f"  Loaded {len(seq)} chronological frames.")

    # 2. Quét các file PCD tương ứng
    print("[2/3] Matching PCD files...")
    pcd_files = sorted(glob.glob(str(pcd_dir / "*.pcd")))
    if not pcd_files:
        print(f"[Error] No .pcd files found in {pcd_dir}")
        sys.exit(1)

    pcd_map = {}
    for rank, cvat_idx, sample_id, t_sec in seq:
        # Thử tìm theo sample_id hoặc cvat_idx
        cand1 = pcd_dir / f"{sample_id}.pcd"
        cand2 = pcd_dir / f"sample_{cvat_idx + 1:02d}.pcd"
        cand3 = pcd_dir / f"frame_{cvat_idx:06d}.pcd"
        if cand1.exists():
            pcd_map[cvat_idx] = str(cand1)
        elif cand2.exists():
            pcd_map[cvat_idx] = str(cand2)
        elif cand3.exists():
            pcd_map[cvat_idx] = str(cand3)
        elif cvat_idx < len(pcd_files):
            pcd_map[cvat_idx] = pcd_files[cvat_idx]

    print(f"  Matched {len(pcd_map)} PCD files.")

    # 3. Khởi tạo Stitcher
    stitcher = SceneStitcher(
        ego_pose_mgr=mgr,
        window_size=args.window,
        voxel_size=args.voxel_size,
        enable_trajectory_smoothing=not args.no_traj_smooth,
        enable_icp=not args.no_icp,
        enable_sor=not args.no_sor
    )

    if args.max_frames:
        stitcher.frame_indices = stitcher.frame_indices[:args.max_frames]

    print(f"[3/3] Running Stitching & Smoothing Pipeline on {len(stitcher.frame_indices)} frames...")

    def progress_callback(cur, total, stats):
        print(
            f"  [{cur:02d}/{total:02d}] Frame #{stats['target_frame']:02d}: "
            f"{stats['original_points']} -> {stats['raw_accumulated_points']} raw -> "
            f"{stats['final_smoothed_points']} smoothed pts | "
            f"ICP RMSE: {stats['mean_icp_rmse']:.4f}m | Time: {stats['elapsed_seconds']:.2f}s"
        )

    results = stitcher.process_sequence(
        pcd_files_map=pcd_map,
        output_dir=args.output_dir,
        binary=args.binary,
        on_progress=progress_callback
    )

    print("------------------------------------------------------------------")
    print(f"[Success] Completed processing {len(results)} frames.")
    print(f"Stitched PCD files saved to: {args.output_dir}")
    print("==================================================================")

if __name__ == "__main__":
    main()
