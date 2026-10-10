import os
import sys
import json
import zipfile
import numpy as np
import open3d as o3d
from pathlib import Path

def generate_lidar_frame(boxes_3d, seed=42):
    """Generate realistic 64-beam automotive LiDAR point cloud."""
    np.random.seed(seed)
    all_points = []
    all_intensities = []

    # 1. Ground Plane (64-beam LiDAR simulation)
    ground_z = -1.65
    beam_angles_v = np.linspace(np.radians(-25), np.radians(2), 48)
    azimuths = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    
    for phi in beam_angles_v:
        if phi >= 0:
            continue
        dist_ground = -ground_z / np.sin(-phi)
        if dist_ground > 75.0 or dist_ground < 1.5:
            continue
        
        r = dist_ground * np.cos(phi)
        x = r * np.cos(azimuths)
        y = r * np.sin(azimuths)
        z = np.full_like(x, ground_z) + np.random.normal(0, 0.015, size=len(x))
        
        intensity = np.full_like(x, 0.15)
        lane_mask = (np.abs(y - 1.75) < 0.12) | (np.abs(y + 1.75) < 0.12) | (np.abs(y - 5.25) < 0.12)
        intensity[lane_mask] = 0.85
        
        pts = np.stack([x, y, z], axis=-1)
        all_points.append(pts)
        all_intensities.append(intensity)

    # 2. Road Curbs & Sidewalks (y = -8.5m and y = +8.5m)
    for curb_y in [-8.5, 8.5]:
        cx = np.linspace(-40, 50, 600)
        cy = np.full_like(cx, curb_y) + np.random.normal(0, 0.05, len(cx))
        cz = np.full_like(cx, ground_z + 0.15) + np.random.normal(0, 0.02, len(cx))
        all_points.append(np.stack([cx, cy, cz], axis=-1))
        all_intensities.append(np.full_like(cx, 0.4))

    # 3. Ambient Environment (Light poles, trees, buildings in background)
    for env_x in [-25, -10, 5, 20, 35]:
        for env_y in [-11.0, 11.0]:
            pz = np.linspace(ground_z, ground_z + 5.0, 40)
            px = np.full_like(pz, env_x) + np.random.normal(0, 0.03, len(pz))
            py = np.full_like(pz, env_y) + np.random.normal(0, 0.03, len(pz))
            all_points.append(np.stack([px, py, pz], axis=-1))
            all_intensities.append(np.full_like(pz, 0.7))

    # 4. Generate points on 3D Bounding Boxes
    for box in boxes_3d:
        center = np.array(box["center"], dtype=np.float32)
        size = np.array(box["size"], dtype=np.float32)
        yaw = float(box.get("rotation_yaw", 0.0))
        label = box.get("label", "Car")

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot_matrix = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ], dtype=np.float32)

        dx, dy, dz = size[0], size[1], size[2]
        num_pts = 350 if label in ["Car", "Bus", "Truck"] else 120
        if label == "Bus": num_pts = 600
        if label == "Truck": num_pts = 500

        local_pts = []
        for sign in [-0.5, 0.5]:
            sx = np.random.uniform(-dx/2, dx/2, num_pts // 5)
            sz = np.random.uniform(-dz/2, dz/2, num_pts // 5)
            sy = np.full_like(sx, sign * dy)
            local_pts.append(np.stack([sx, sy, sz], axis=-1))
        for sign in [-0.5, 0.5]:
            sy = np.random.uniform(-dy/2, dy/2, num_pts // 5)
            sz = np.random.uniform(-dz/2, dz/2, num_pts // 5)
            sx = np.full_like(sy, sign * dx)
            local_pts.append(np.stack([sx, sy, sz], axis=-1))
        rx = np.random.uniform(-dx/2, dx/2, num_pts // 5)
        ry = np.random.uniform(-dy/2, dy/2, num_pts // 5)
        rz = np.full_like(rx, dz/2)
        local_pts.append(np.stack([rx, ry, rz], axis=-1))

        box_local = np.concatenate(local_pts, axis=0)
        box_local += np.random.normal(0, 0.02, size=box_local.shape)
        box_world = (box_local @ rot_matrix.T) + center
        
        all_points.append(box_world)
        intensity_val = 0.75 if label in ["Car", "Bus", "Truck"] else 0.35
        all_intensities.append(np.full(len(box_world), intensity_val))

    points_concat = np.concatenate(all_points, axis=0).astype(np.float32)
    intensities_concat = np.concatenate(all_intensities, axis=0).astype(np.float32)

    dists = np.linalg.norm(points_concat[:, :2], axis=1)
    mask = (dists >= 1.0) & (dists <= 90.0)
    points_clean = points_concat[mask]
    intensities_clean = intensities_concat[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_clean.astype(np.float64))
    
    intensity_norm = np.clip(intensities_clean, 0.0, 1.0)
    colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=-1)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd

def interpolate_boxes(b_prev, b_next, alpha=0.5):
    """Interpolate between two 3D boxes."""
    c_prev = np.array(b_prev["center"])
    c_next = np.array(b_next["center"])
    s_prev = np.array(b_prev["size"])
    s_next = np.array(b_next["size"])
    yaw_prev = float(b_prev.get("rotation_yaw", 0.0))
    yaw_next = float(b_next.get("rotation_yaw", 0.0))
    
    # Handle angle wrapping for yaw
    diff = (yaw_next - yaw_prev + np.pi) % (2 * np.pi) - np.pi
    yaw_interp = yaw_prev + alpha * diff
    
    return {
        "label": b_prev["label"],
        "center": (c_prev + alpha * (c_next - c_prev)).tolist(),
        "size": (s_prev + alpha * (s_next - s_prev)).tolist(),
        "rotation_yaw": float(yaw_interp)
    }

def main():
    task_dir = Path("/media/bang/Work3/projects/cvat/cvat-annotator-dashboard-3d/data/cvat_tasks/000201")
    gt_file = task_dir / "ground_truth" / "gt_boxes3d.json"
    pcd_dir = task_dir / "point_cloud"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    
    with open(gt_file, "r") as f:
        gt_manifest = json.load(f)
        
    num_frames = len(gt_manifest)
    print(f"[*] Processing {num_frames} frames for Sequence 000201...")
    
    # 1. Fill in missing odd frames via interpolation
    for i in range(1, num_frames, 2):
        prev_boxes = gt_manifest[i - 1]["boxes_3d"]
        next_boxes = gt_manifest[i + 1]["boxes_3d"] if i + 1 < num_frames else prev_boxes
        
        interp_boxes = []
        matched_next = set()
        for pb in prev_boxes:
            c1 = np.array(pb["center"])
            best_dist = 3.5
            best_nb = None
            best_idx = None
            for idx_n, nb in enumerate(next_boxes):
                if idx_n in matched_next or nb["label"] != pb["label"]:
                    continue
                c2 = np.array(nb["center"])
                d = np.linalg.norm(c1 - c2)
                if d < best_dist:
                    best_dist = d
                    best_nb = nb
                    best_idx = idx_n
            if best_nb is not None:
                matched_next.add(best_idx)
                interp_boxes.append(interpolate_boxes(pb, best_nb, 0.5))
            else:
                interp_boxes.append(pb)
                
        gt_manifest[i]["boxes_3d"] = interp_boxes
        gt_manifest[i]["objects_count"] = len(interp_boxes)
        
    with open(gt_file, "w") as f:
        json.dump(gt_manifest, f, indent=2)
    print(f"[+] All {num_frames} frames now have continuous 3D bounding box trajectories!")
    
    # 2. Generate .pcd files
    print(f"[*] Generating {num_frames} 64-beam LiDAR point clouds (.pcd)...")
    for i, item in enumerate(gt_manifest):
        pcd = generate_lidar_frame(item["boxes_3d"], seed=200 + i)
        out_pcd = pcd_dir / f"{i:06d}.pcd"
        o3d.io.write_point_cloud(str(out_pcd), pcd)
        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"    ✓ Frame {i+1}/{num_frames}: {out_pcd.name} ({len(pcd.points)} points, {item['objects_count']} objects)")
            
    # 3. Create ready-to-upload CVAT 3D zip package
    zip_path = task_dir.parent / "000201_39frames_cvat.zip"
    print(f"\n[*] Packaging complete CVAT 3D dataset into: {zip_path.name}...")
    
    related_dir = task_dir / "related_images"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for pcd_file in sorted(pcd_dir.glob("*.pcd")):
            z.write(pcd_file, arcname=f"pointcloud/{pcd_file.name}")
        for frame_folder in sorted(related_dir.glob("*_pcd")):
            for img in sorted(frame_folder.glob("*.jpg")):
                z.write(img, arcname=f"related_images/{frame_folder.name}/{img.name}")
                
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully packaged {zip_path.name} ({zip_size_mb:.2f} MB)")
    print("[+] Ready for CVAT upload!")

if __name__ == "__main__":
    main()
