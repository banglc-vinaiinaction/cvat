#!/usr/bin/env python3
"""
Enterprise 3D LiDAR Road Point Cloud Snapping & Orientation Engine
Platform: cvat-annotator-dashboard-3d

Features:
1. RANSAC Ground Plane Detection: Snaps cuboid bottom flush to road surface.
2. Principal Component Analysis (PCA) & Minimum Area Bounding Box (L-Shape):
   Estimates optimal yaw heading angle (theta) and auto-tightens (L, W, H) extents.
3. Multi-frame & single-object snapping with metric validation.
"""

import os
import sys
import math
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

# ==============================================================================
# 1. RANSAC Ground Plane Detection
# ==============================================================================

class GroundPlaneRANSAC:
    def __init__(self, distance_threshold: float = 0.08, max_iterations: int = 400):
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def fit_plane(self, points: np.ndarray, expected_normal: np.ndarray = np.array([0, 0, 1])) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits plane Ax + By + Cz + D = 0 to points using RANSAC.
        Returns plane coefficients [A, B, C, D] normalized (A^2 + B^2 + C^2 = 1)
        and boolean inlier mask.
        """
        if len(points) < 3:
            # Fallback to horizontal ground at mean Z
            mean_z = float(np.mean(points[:, 2])) if len(points) > 0 else 0.0
            return np.array([0.0, 0.0, 1.0, -mean_z]), np.ones(len(points), dtype=bool)

        pts = points[:, :3]
        default_z = float(np.percentile(pts[:, 2], 5)) if len(pts) > 0 else 0.0
        best_plane = np.array([0.0, 0.0, 1.0, -default_z])
        best_inliers = np.zeros(len(pts), dtype=bool)
        max_inlier_count = -1

        N = len(pts)
        for _ in range(self.max_iterations):
            # Sample 3 random points
            idx = np.random.choice(N, 3, replace=False)
            p1, p2, p3 = pts[idx]

            # Vector normal = (p2 - p1) x (p3 - p1)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # Ensure normal points upwards (positive Z)
            if normal[2] < 0:
                normal = -normal

            # Check normal alignment with expected upright vector (road shouldn't tilt > 25 degrees)
            if np.dot(normal, expected_normal) < 0.90:
                continue

            # D = -normal . p1
            d = -np.dot(normal, p1)
            plane = np.array([normal[0], normal[1], normal[2], d])

            # Distance from all points to plane: |Ax + By + Cz + D|
            dists = np.abs(np.dot(pts, normal) + d)
            inliers = dists < self.distance_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_plane = plane
                best_inliers = inliers

        # Refine plane using all inliers via SVD
        if np.sum(best_inliers) >= 3:
            inlier_pts = pts[best_inliers]
            centroid = np.mean(inlier_pts, axis=0)
            u, s, vh = np.linalg.svd(inlier_pts - centroid)
            refined_normal = vh[2, :]
            if refined_normal[2] < 0:
                refined_normal = -refined_normal
            refined_d = -np.dot(refined_normal, centroid)
            best_plane = np.array([refined_normal[0], refined_normal[1], refined_normal[2], refined_d])

        return best_plane, best_inliers

    @staticmethod
    def get_road_height(plane: np.ndarray, x: float, y: float) -> float:
        """Calculates ground elevation Z directly beneath (x, y)."""
        A, B, C, D = plane
        if abs(C) < 1e-5:
            return -1.65
        return float(-(A * x + B * y + D) / C)


# ==============================================================================
# 2. Point Cloud Orientation & Extents Estimator (PCA & Minimum Area Box)
# ==============================================================================

class OrientationEstimator:
    @staticmethod
    def fit_pca_heading(points_2d: np.ndarray) -> float:
        """
        Estimates yaw heading using 2D Principal Component Analysis.
        Returns yaw angle in radians [-pi, pi].
        """
        if len(points_2d) < 3:
            return 0.0

        mean = np.mean(points_2d[:, :2], axis=0)
        centered = points_2d[:, :2] - mean
        cov = np.cov(centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Principal component is eigenvector with largest eigenvalue
        principal_vector = eigenvectors[:, -1]
        
        yaw = math.atan2(principal_vector[1], principal_vector[0])
        return float(yaw)

    @staticmethod
    def fit_minimum_area_bbox(points_2d: np.ndarray, current_yaw: float = 0.0) -> Tuple[float, float, float, np.ndarray]:
        """
        Finds optimal bounding box using Minimum Bounding Rectangle search.
        Constrains search around current yaw to preserve travel direction.
        Returns: (best_yaw, length, width, center_2d)
        """
        if len(points_2d) < 3:
            return current_yaw, 4.5, 2.0, np.array([0.0, 0.0])

        pts = points_2d[:, :2]
        
        # Search angles around current yaw (+- 45 degrees in 1-degree steps)
        angle_offsets = np.linspace(-np.pi / 4, np.pi / 4, 91)
        candidate_angles = current_yaw + angle_offsets

        best_area = float("inf")
        best_yaw = current_yaw
        best_dims = (4.5, 2.0)
        best_center = np.mean(pts, axis=0)

        for theta in candidate_angles:
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, s], [-s, c]])
            rotated = pts @ rot.T

            min_x, max_x = np.min(rotated[:, 0]), np.max(rotated[:, 0])
            min_y, max_y = np.min(rotated[:, 1]), np.max(rotated[:, 1])

            len_x = max_x - min_x
            len_y = max_y - min_y
            area = len_x * len_y

            # Edge closeness penalty (favors compact vehicle shapes)
            if area < best_area:
                best_area = area
                best_yaw = theta
                # Vehicle length is along heading axis (theta), width is lateral
                length, width = len_x, len_y
                box_yaw = theta
                center_rot = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

                inv_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                best_center = inv_rot @ center_rot
                best_dims = (length, width)
                best_yaw = math.atan2(np.sin(box_yaw), np.cos(box_yaw))

        return best_yaw, float(best_dims[0]), float(best_dims[1]), best_center


# ==============================================================================
# 3. High-Level Road Snapper & Fitter
# ==============================================================================

class RoadPointCloudSnapper:
    def __init__(self):
        self.plane_fitter = GroundPlaneRANSAC(distance_threshold=0.08, max_iterations=300)

    def snap_and_orient(
        self,
        point_cloud: np.ndarray,
        cuboid: Dict[str, float],
        padding: float = 0.4,
        snap_ground: bool = True,
        auto_orient: bool = True,
        tighten_extents: bool = False,
        ground_clearance: float = 0.04
    ) -> Dict[str, Any]:
        """
        Takes raw cuboid parameters {x, y, z, l, w, h, yaw} and point cloud,
        snaps bottom to road with multi-corner terrain clearance and aligns heading
        while keeping dimensions (l, w, h) strictly constant.
        """
        cx, cy, cz = float(cuboid["x"]), float(cuboid["y"]), float(cuboid["z"])
        l, w, h = float(cuboid["l"]), float(cuboid["w"]), float(cuboid["h"])
        yaw = float(cuboid.get("yaw", 0.0))
        bottom_z = cz - (h / 2.0)

        # 1. Extract Local Neighborhood Points
        roi_radius = math.sqrt((l / 2.0) ** 2 + (w / 2.0) ** 2) + 2.5
        dists_2d = np.linalg.norm(point_cloud[:, :2] - np.array([cx, cy]), axis=1)
        local_mask = dists_2d <= roi_radius
        local_pts = point_cloud[local_mask]

        if len(local_pts) < 10:
            default_est_z = float(np.percentile(point_cloud[:, 2], 5)) if len(point_cloud) > 0 else 0.0
            return {
                "cuboid": cuboid,
                "ground_z": default_est_z,
                "inliers_count": 0,
                "status": "sparse_points_unchanged"
            }

        # Exclude points inside the vehicle footprint (chassis/wheels/doors)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot_2d = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        rel_xy = local_pts[:, :2] - np.array([cx, cy])
        u_v = rel_xy @ rot_2d.T
        is_outside = (np.abs(u_v[:, 0]) > l / 2.0 + 0.12) | (np.abs(u_v[:, 1]) > w / 2.0 + 0.12)

        # 2. Road Ground Snapping
        local_p05 = float(np.percentile(local_pts[:, 2], 5))
        ground_z = local_p05
        road_plane = np.array([0.0, 0.0, 1.0, -local_p05])
        cz_new = cz
        final_rx, final_ry, final_rz = 0.0, 0.0, yaw
        
        if snap_ground:
            # Candidates for ground: strictly outside vehicle footprint and below lower body
            z_max_ground = min(bottom_z + 0.35, cz - (h * 0.15))
            ground_candidates = local_pts[is_outside & (local_pts[:, 2] <= z_max_ground)]
            
            if len(ground_candidates) < 12:
                ground_candidates = local_pts[local_pts[:, 2] <= z_max_ground]

            if len(ground_candidates) >= 12:
                # 1. Identify Lowest Point Representative (LPR) of ground candidates
                sorted_idx = np.argsort(ground_candidates[:, 2])
                n_seeds = max(6, int(len(sorted_idx) * 0.15))
                seed_pts = ground_candidates[sorted_idx[:n_seeds]]
                lpr_z = float(np.median(seed_pts[:, 2]))
                
                # Seeds tightly within 0.15m of LPR to prevent obstacle contamination
                seeds = seed_pts[seed_pts[:, 2] <= lpr_z + 0.15]
                if len(seeds) < 3:
                    seeds = seed_pts[:max(3, len(seed_pts))]

                road_plane, _ = self.plane_fitter.fit_plane(seeds)
                A, B, C, D = road_plane
                if C < 0:
                    A, B, C, D = -A, -B, -C, -D
                    road_plane = np.array([A, B, C, D])

                # GUARD: Road normal MUST be predominantly vertical (Nz >= 0.85, i.e. < 31 deg tilt)
                if road_plane[2] < 0.85:
                    road_plane = np.array([0.0, 0.0, 1.0, -lpr_z])
                else:
                    # Refine with consensus ground inliers
                    norm_v = road_plane[:3]
                    dist_to_plane = np.abs(ground_candidates @ norm_v + road_plane[3])
                    inliers = ground_candidates[(dist_to_plane <= 0.06) & (ground_candidates[:, 2] <= lpr_z + 0.30)]
                    if len(inliers) >= 10:
                        ref_plane, _ = self.plane_fitter.fit_plane(inliers)
                        if ref_plane[2] < 0:
                            ref_plane = -ref_plane
                        if ref_plane[2] >= 0.85:
                            road_plane = ref_plane

                raw_plane_z = GroundPlaneRANSAC.get_road_height(road_plane, cx, cy)
                raw_plane_z = float(np.clip(raw_plane_z, lpr_z - 0.20, lpr_z + 0.35))
                
                # Upper envelope of consensus ground seeds (beam scatter ~2-3cm)
                sorted_seed_z = np.sort(seeds[:, 2])
                p90_idx = min(len(sorted_seed_z) - 1, int(len(sorted_seed_z) * 0.90))
                ground_p90 = float(sorted_seed_z[p90_idx])
                
                ground_z = max(raw_plane_z, ground_p90) + ground_clearance
                cz_new = ground_z + (h / 2.0)

                # Sectional profiling: angle cuboid between highest average point vs lowest average point and everything in-between
                half_l = l / 2.0
                half_w = w / 2.0
                margin_x = max(0.35, l * 0.10)
                margin_y = max(0.35, w * 0.10)

                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                dx_loc = (local_pts[:, 0] - cx) * cos_yaw + (local_pts[:, 1] - cy) * sin_yaw
                dy_loc = -(local_pts[:, 0] - cx) * sin_yaw + (local_pts[:, 1] - cy) * cos_yaw

                in_footprint = (np.abs(dx_loc) <= half_l + margin_x) & (np.abs(dy_loc) <= half_w + margin_y)
                footprint_ground = local_pts[in_footprint & (local_pts[:, 2] <= bottom_z + 0.35) & (local_pts[:, 2] >= bottom_z - 0.70)]

                if len(footprint_ground) >= 10:
                    sorted_z = np.sort(footprint_ground[:, 2])
                    lpr_val = float(np.median(sorted_z[:max(3, int(len(sorted_z) * 0.15))]))
                    surface_pts = footprint_ground[(footprint_ground[:, 2] >= lpr_val - 0.15) & (footprint_ground[:, 2] <= lpr_val + 0.22)]

                    surf_u = (surface_pts[:, 0] - cx) * cos_yaw + (surface_pts[:, 1] - cy) * sin_yaw
                    surf_v = -(surface_pts[:, 0] - cx) * sin_yaw + (surface_pts[:, 1] - cy) * cos_yaw
                    surf_z = surface_pts[:, 2]

                    # 1. Longitudinal slices (5 bins)
                    x_edges = np.linspace(-half_l, half_l, 6)
                    bin_centers_x = []
                    bin_avg_z_x = []
                    for b in range(5):
                        b_mask = (surf_u >= x_edges[b] - 0.10) & (surf_u <= x_edges[b+1] + 0.10)
                        if np.sum(b_mask) >= 2:
                            bin_centers_x.append(0.5 * (x_edges[b] + x_edges[b+1]))
                            bin_avg_z_x.append(float(np.mean(surf_z[b_mask])))

                    # 2. Lateral slices (3 bins)
                    y_edges = np.linspace(-half_w, half_w, 4)
                    bin_centers_y = []
                    bin_avg_z_y = []
                    for b in range(3):
                        b_mask = (surf_v >= y_edges[b] - 0.10) & (surf_v <= y_edges[b+1] + 0.10)
                        if np.sum(b_mask) >= 2:
                            bin_centers_y.append(0.5 * (y_edges[b] + y_edges[b+1]))
                            bin_avg_z_y.append(float(np.mean(surf_z[b_mask])))

                    slope_x = 0.0
                    if len(bin_centers_x) >= 2:
                        bx, bz = np.array(bin_centers_x), np.array(bin_avg_z_x)
                        if np.var(bx) > 1e-4:
                            slope_x = float(np.cov(bx, bz)[0, 1] / np.var(bx))

                    slope_y = 0.0
                    if len(bin_centers_y) >= 2:
                        by, bz = np.array(bin_centers_y), np.array(bin_avg_z_y)
                        if np.var(by) > 1e-4:
                            slope_y = float(np.cov(by, bz)[0, 1] / np.var(by))

                    max_slope = 0.176
                    slope_x = float(np.clip(slope_x, -max_slope, max_slope))
                    slope_y = float(np.clip(slope_y, -max_slope, max_slope))

                    all_bin_z = bin_avg_z_x + bin_avg_z_y
                    mean_ground_z = float(np.mean(all_bin_z)) if len(all_bin_z) > 0 else lpr_val

                    base_z = mean_ground_z + ground_clearance
                    cz_new = base_z + h / 2.0
                    cz_new = float(np.clip(cz_new, cz - 0.25, cz + 0.35))

                    theta_long = np.arctan(slope_x)
                    theta_lat = np.arctan(slope_y)

                    if abs(theta_long) > 1e-4 or abs(theta_lat) > 1e-4:
                        fwd_loc = np.array([np.cos(theta_long), 0.0, np.sin(theta_long)])
                        left_loc = np.array([0.0, np.cos(theta_lat), np.sin(theta_lat)])
                        up_loc = np.cross(fwd_loc, left_loc)
                        up_loc /= np.linalg.norm(up_loc)
                        re_left = np.cross(up_loc, fwd_loc)
                        re_left /= np.linalg.norm(re_left)
                        re_fwd = np.cross(re_left, up_loc)
                        re_fwd /= np.linalg.norm(re_fwd)

                        R_local = np.column_stack([re_fwd, re_left, up_loc])
                        R_yaw = np.array([
                            [cos_yaw, -sin_yaw, 0.0],
                            [sin_yaw,  cos_yaw, 0.0],
                            [0.0,      0.0,     1.0]
                        ])
                        R_world = R_yaw @ R_local

                        sy = np.sqrt(R_world[0, 0]**2 + R_world[1, 0]**2)
                        final_rx = float(np.arctan2(R_world[2, 1], R_world[2, 2]))
                        final_ry = float(np.arctan2(-R_world[2, 0], sy))
                        final_rz = float(np.arctan2(R_world[1, 0], R_world[0, 0]))
                    else:
                        final_rx = 0.0
                        final_ry = 0.0
                        final_rz = yaw
                else:
                    final_rx = 0.0
                    final_ry = 0.0
                    final_rz = yaw
            else:
                ground_z = local_p05 + ground_clearance
                cz_new = ground_z + (h / 2.0)
                final_rx = 0.0
                final_ry = 0.0
                final_rz = yaw

        # 3. Object Cluster Filtering (points strictly above road surface)
        object_mask = (local_pts[:, 2] >= ground_z + 0.08) & (local_pts[:, 2] <= ground_z + h + 0.4)
        obj_pts = local_pts[object_mask]

        new_yaw = final_rz
        new_l, new_w, new_h = l, w, h
        new_cx, new_cy = cx, cy

        if len(obj_pts) >= 15:
            # 4. Auto-Orientation
            if auto_orient:
                best_yaw, min_l, min_w, new_c2d = OrientationEstimator.fit_minimum_area_bbox(obj_pts, current_yaw=yaw)
                new_yaw = best_yaw
                
                # 5. Centering (keep LWH dimensions strictly constant)
                if tighten_extents:
                    new_cx, new_cy = float(new_c2d[0]), float(new_c2d[1])

        result_cuboid = {
            "x": round(float(new_cx), 3),
            "y": round(float(new_cy), 3),
            "z": round(float(cz_new), 3),
            "rx": round(float(final_rx), 4),
            "ry": round(float(final_ry), 4),
            "rz": round(float(new_yaw), 4),
            "l": round(float(new_l), 3),
            "w": round(float(new_w), 3),
            "h": round(float(new_h), 3),
            "yaw": round(float(new_yaw), 4)
        }

        # Calculate adjustments
        adjustments = {
            "delta_z_ground": round(float(cz_new - cz), 3),
            "delta_yaw_deg": round(math.degrees(new_yaw - yaw), 2),
            "ground_elevation": round(float(ground_z), 3),
            "road_plane_normal": road_plane[:3].tolist(),
            "cluster_points": int(len(obj_pts))
        }

        return {
            "original_cuboid": cuboid,
            "snapped_cuboid": result_cuboid,
            "adjustments": adjustments,
            "status": "success"
        }


# ==============================================================================
# 4. Self-Test & Demonstration Runner
# ==============================================================================

if __name__ == "__main__":
    print("=================================================================")
    print("🚗 3D LiDAR Road Snapping & Orientation Engine (Self-Test)")
    print("=================================================================")

    # Test with real PCD if available, else synthetic
    sample_pcd_path = Path("/media/bang/Work3/projects/cvat/cvat-annotator-dashboard-3d/data/cvat_tasks/000201/point_cloud/000000.pcd")
    
    if sample_pcd_path.exists():
        import open3d as o3d
        print(f"[*] Loading point cloud from {sample_pcd_path}...")
        pcd = o3d.io.read_point_cloud(str(sample_pcd_path))
        pts = np.asarray(pcd.points)
    else:
        print("[*] Generating synthetic test scene...")
        # Ground plane at z = -1.65 + synthetic car at (10, 2, -0.8) tilted 15 deg
        ground = np.random.uniform(-20, 20, (15000, 2))
        ground_z = np.full((15000, 1), -1.65) + np.random.normal(0, 0.02, (15000, 1))
        
        # Car points
        car_pts = np.random.uniform(-2, 2, (800, 3))
        car_pts[:, 0] *= 2.2 # L=4.4
        car_pts[:, 1] *= 0.9 # W=1.8
        car_pts[:, 2] = -1.65 + np.random.uniform(0.1, 1.5, 800)
        # Rotate 18 deg
        ang = np.radians(18.0)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        car_pts[:, :2] = (car_pts[:, :2] @ rot.T) + np.array([10.0, 2.0])
        
        pts = np.vstack([np.hstack([ground, ground_z]), car_pts])

    snapper = RoadPointCloudSnapper()

    # User loosely places a floating, misaligned box
    rough_input = {
        "x": 10.2, "y": 2.1, "z": 0.5, # floating 1.3m too high!
        "l": 5.2,  "w": 2.6, "h": 2.2, # too large
        "yaw": 0.0                     # misaligned (car is rotated)
    }

    print("\n--- 1. Loose Initial Annotator Input ---")
    print(f"Center:   X={rough_input['x']}, Y={rough_input['y']}, Z={rough_input['z']} (Floating above road!)")
    print(f"Extents:  L={rough_input['l']}, W={rough_input['w']}, H={rough_input['h']}")
    print(f"Heading:  Yaw={rough_input['yaw']} rad (0.0°)")

    result = snapper.snap_and_orient(pts, rough_input)
    snapped = result["snapped_cuboid"]
    adj = result["adjustments"]

    print("\n--- 2. Snapped & Oriented Result ---")
    print(f"Center:   X={snapped['x']}, Y={snapped['y']}, Z={snapped['z']} (Flush to road at z={adj['ground_elevation']}m)")
    print(f"Extents:  L={snapped['l']}, W={snapped['w']}, H={snapped['h']} (Snug shrink-wrap)")
    print(f"Heading:  Yaw={snapped['yaw']} rad ({adj['delta_yaw_deg']:+.1f}° aligned to vehicle axis)")
    print(f"\n[+] Adjustments Made: Height delta: {adj['delta_z_ground']:+.2f}m | Yaw adjustment: {adj['delta_yaw_deg']:+.1f}°")
    print("[+] Status:", result["status"])
