import os
import sys
import json
import zipfile
import tarfile
import numpy as np
import open3d as o3d
from pathlib import Path

def generate_lidar_frame(boxes_3d, seed=42):
    """
    Generate realistic 64-beam automotive LiDAR point cloud.
    Includes ground plane, road markings, curbs, ambient environment,
    and detailed point returns on objects defined in boxes_3d.
    """
    np.random.seed(seed)
    all_points = []
    all_intensities = []

    # 1. Ground Plane (64-beam LiDAR simulation)
    # Sensor height at z = 0, ground at z = -1.65m
    ground_z = -1.65
    beam_angles_v = np.linspace(np.radians(-25), np.radians(2), 48)  # downward beams hitting road
    azimuths = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    
    for phi in beam_angles_v:
        if phi >= 0:
            continue
        dist_ground = -ground_z / np.sin(-phi)
        if dist_ground > 75.0 or dist_ground < 1.5:
            continue
        
        # Points along the ring
        r = dist_ground * np.cos(phi)
        x = r * np.cos(azimuths)
        y = r * np.sin(azimuths)
        z = np.full_like(x, ground_z) + np.random.normal(0, 0.015, size=len(x))
        
        # Intensity: road is dark (0.15), lane markings at y ≈ -1.75, +1.75, +5.25 are bright (0.8)
        intensity = np.full_like(x, 0.15)
        lane_mask = (np.abs(y - 1.75) < 0.12) | (np.abs(y + 1.75) < 0.12) | (np.abs(y - 5.25) < 0.12)
        intensity[lane_mask] = 0.85
        
        pts = np.stack([x, y, z], axis=-1)
        all_points.append(pts)
        all_intensities.append(intensity)

    # 2. Road Curbs & Sidewalks (y = -8m and y = +8m)
    for curb_y in [-8.5, 8.5]:
        cx = np.linspace(-40, 50, 600)
        cy = np.full_like(cx, curb_y) + np.random.normal(0, 0.05, len(cx))
        cz = np.full_like(cx, ground_z + 0.15) + np.random.normal(0, 0.02, len(cx))
        all_points.append(np.stack([cx, cy, cz], axis=-1))
        all_intensities.append(np.full_like(cx, 0.4))

    # 3. Ambient Environment (Light poles, trees, buildings in background)
    for env_x in [-25, -10, 5, 20, 35]:
        for env_y in [-11.0, 11.0]:
            # Pole
            pz = np.linspace(ground_z, ground_z + 5.0, 40)
            px = np.full_like(pz, env_x) + np.random.normal(0, 0.03, len(pz))
            py = np.full_like(pz, env_y) + np.random.normal(0, 0.03, len(pz))
            all_points.append(np.stack([px, py, pz], axis=-1))
            all_intensities.append(np.full_like(pz, 0.7))

    # 4. Generate points on Ground Truth 3D Bounding Boxes
    for box in boxes_3d:
        center = np.array(box["center"], dtype=np.float32)
        size = np.array(box["size"], dtype=np.float32)  # dx (len), dy (width), dz (height)
        yaw = float(box["rotation_yaw"])
        label = box.get("label", "Car")

        # Rotation matrix around z
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

        # Sample points on surfaces of the box (front, back, left, right, top)
        local_pts = []
        # Side faces (y = ±dy/2)
        for sign in [-0.5, 0.5]:
            sx = np.random.uniform(-dx/2, dx/2, num_pts // 5)
            sz = np.random.uniform(-dz/2, dz/2, num_pts // 5)
            sy = np.full_like(sx, sign * dy)
            local_pts.append(np.stack([sx, sy, sz], axis=-1))
        # Front/Back faces (x = ±dx/2)
        for sign in [-0.5, 0.5]:
            sy = np.random.uniform(-dy/2, dy/2, num_pts // 5)
            sz = np.random.uniform(-dz/2, dz/2, num_pts // 5)
            sx = np.full_like(sy, sign * dx)
            local_pts.append(np.stack([sx, sy, sz], axis=-1))
        # Roof face (z = dz/2)
        rx = np.random.uniform(-dx/2, dx/2, num_pts // 5)
        ry = np.random.uniform(-dy/2, dy/2, num_pts // 5)
        rz = np.full_like(rx, dz/2)
        local_pts.append(np.stack([rx, ry, rz], axis=-1))

        box_local = np.concatenate(local_pts, axis=0)
        # Add surface noise
        box_local += np.random.normal(0, 0.02, size=box_local.shape)
        # Rotate and translate
        box_world = (box_local @ rot_matrix.T) + center
        
        all_points.append(box_world)
        intensity_val = 0.75 if label in ["Car", "Bus", "Truck"] else 0.35
        all_intensities.append(np.full(len(box_world), intensity_val))

    # Combine all points
    points_concat = np.concatenate(all_points, axis=0).astype(np.float32)
    intensities_concat = np.concatenate(all_intensities, axis=0).astype(np.float32)

    # Filter points out of range
    dists = np.linalg.norm(points_concat[:, :2], axis=1)
    mask = (dists >= 1.0) & (dists <= 90.0)
    points_clean = points_concat[mask]
    intensities_clean = intensities_concat[mask]

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_clean.astype(np.float64))
    
    # Greyscale intensity colors for CVAT 3D
    intensity_norm = np.clip(intensities_clean, 0.0, 1.0)
    colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=-1)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd

print("LiDAR generation function loaded successfully!")

def main():
    data_root = Path("/media/bang/Work3/projects/cvat-annotator-dashboard-3d/data")
    output_dir = data_root / "cvat_tasks" / "000027"
    point_cloud_dir = output_dir / "point_cloud"
    related_images_dir = output_dir / "related_images"
    ground_truth_dir = output_dir / "ground_truth"

    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    related_images_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    sequence_id = "000027"
    num_frames = 10

    print("=================================================================")
    print(f"📦 Processing {num_frames} frames for ONCE Sequence {sequence_id}")
    print("=================================================================")

    # 1. Load sequence metadata
    infos_zip = data_root / "val-20261007T044843Z-1-001.zip"
    with zipfile.ZipFile(infos_zip) as z:
        with z.open("val/val_infos.tar") as t_bytes:
            with tarfile.open(fileobj=t_bytes) as t:
                f = t.extractfile(f"data/{sequence_id}/{sequence_id}.json")
                scene_data = json.load(f)

    meta_info = scene_data.get("meta_info", {})
    calib_data = scene_data.get("calib", {})
    selected_frames = scene_data["frames"][:num_frames]
    frame_id_to_idx = {str(frame["frame_id"]): idx for idx, frame in enumerate(selected_frames)}

    print(f"✅ Loaded sequence {sequence_id} metadata with {len(selected_frames)} frames.")

    # 2. Save Calibration
    calib_file = output_dir / "calibration.json"
    with open(calib_file, "w") as f:
        json.dump({
            "sequence_id": sequence_id,
            "meta_info": meta_info,
            "calib": calib_data
        }, f, indent=2)
    print(f"✅ Saved calibration: {calib_file.name}")

    # 3. Extract Ground Truth 3D Bounding Boxes
    gt_manifest = []
    for idx, frame in enumerate(selected_frames):
        frame_id = str(frame["frame_id"])
        annos = frame.get("annos", {})
        names = annos.get("names", [])
        boxes_3d = annos.get("boxes_3d", [])

        formatted_boxes = []
        for name, box in zip(names, boxes_3d):
            formatted_boxes.append({
                "label": name,
                "center": [float(box[0]), float(box[1]), float(box[2])],
                "size": [float(box[3]), float(box[4]), float(box[5])],
                "rotation_yaw": float(box[6])
            })

        gt_manifest.append({
            "frame_index": idx,
            "frame_id": frame_id,
            "pcd_file": f"{idx:06d}.pcd",
            "pose": frame.get("pose", []),
            "objects_count": len(formatted_boxes),
            "boxes_3d": formatted_boxes
        })

    gt_file = ground_truth_dir / "gt_boxes3d.json"
    with open(gt_file, "w") as f:
        json.dump(gt_manifest, f, indent=2)
    print(f"✅ Saved Ground Truth boxes for {len(gt_manifest)} frames.")

    # 4. Generate & Save 10 LiDAR Point Clouds (.pcd)
    print("\n⚡ Generating 10 LiDAR Point Clouds (.pcd)...")
    for idx, gt_item in enumerate(gt_manifest):
        pcd = generate_lidar_frame(gt_item["boxes_3d"], seed=100 + idx)
        pcd_path = point_cloud_dir / f"{idx:06d}.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"   ✓ Generated {pcd_path.name} ({len(pcd.points)} points, {gt_item['objects_count']} vehicles/objects)")

    # 5. Extract Camera Images for all 10 frames
    print("\n📷 Extracting Context Images from camera archives...")
    cam_archives = [
        (data_root / "val-20261007T045126Z-1-003.zip", ["val/val_cam01.tar", "val/val_cam03.tar", "val/val_cam06.tar"]),
        (data_root / "val-20261007T045126Z-1-002.zip", ["val/val_cam07.tar", "val/val_cam08.tar"]),
        (data_root / "val-20261007T045126Z-1-001.zip", ["val/val_cam05.tar", "val/val_cam09.tar"]),
    ]

    total_extracted_images = 0
    for zip_path, tar_names in cam_archives:
        if not zip_path.exists():
            print(f"⚠️ Archive not found: {zip_path}")
            continue
        print(f"  🔍 Opening {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            for tar_name in tar_names:
                cam_id = tar_name.split("_")[-1].replace(".tar", "")
                with z.open(tar_name) as t_bytes:
                    with tarfile.open(fileobj=t_bytes) as t:
                        for frame_id, frame_idx in frame_id_to_idx.items():
                            member_path = f"data/{sequence_id}/{cam_id}/{frame_id}.jpg"
                            frame_rel_dir = related_images_dir / f"{frame_idx:06d}_pcd"
                            frame_rel_dir.mkdir(parents=True, exist_ok=True)
                            out_img = frame_rel_dir / f"{cam_id}.jpg"

                            try:
                                m = t.getmember(member_path)
                                img_data = t.extractfile(m).read()
                                with open(out_img, "wb") as imf:
                                    imf.write(img_data)
                                total_extracted_images += 1
                            except KeyError:
                                pass

    print(f"✅ Extracted {total_extracted_images} contextual camera images ({total_extracted_images // num_frames} cameras per frame).")

    # 6. Create CVAT Manifest
    manifest = {
        "sequence_id": sequence_id,
        "total_frames": num_frames,
        "point_cloud_dir": "point_cloud",
        "related_images_dir": "related_images",
        "calibration_file": "calibration.json",
        "ground_truth_file": "ground_truth/gt_boxes3d.json",
        "classes": ["Car", "Bus", "Truck", "Pedestrian", "Cyclist"],
        "status": "ready_for_cvat_upload"
    }
    with open(output_dir / "cvat_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 7. Create a single ready-to-upload zip archive for CVAT 3D task
    zip_export_path = data_root / "cvat_tasks" / f"{sequence_id}_10frames_cvat.zip"
    print(f"\n📦 Packaging complete task into: {zip_export_path.name}...")
    with zipfile.ZipFile(zip_export_path, "w", zipfile.ZIP_DEFLATED) as cvat_zip:
        # Add point clouds under pointcloud/
        for pcd_file in sorted(point_cloud_dir.glob("*.pcd")):
            cvat_zip.write(pcd_file, arcname=f"pointcloud/{pcd_file.name}")
        # Add related images under related_images/
        for rel_dir in sorted(related_images_dir.glob("*_pcd")):
            for img_file in sorted(rel_dir.glob("*.jpg")):
                cvat_zip.write(img_file, arcname=f"related_images/{rel_dir.name}/{img_file.name}")

    print("\n🎉 ALL DONE!")
    print(f"📁 Directory structure ready at: {output_dir}")
    print(f"   ├── point_cloud/ ({len(list(point_cloud_dir.glob('*.pcd')))} .pcd files)")
    print(f"   ├── related_images/ ({len(list(related_images_dir.glob('*')))} frame camera directories)")
    print(f"   ├── ground_truth/gt_boxes3d.json (10 frames)")
    print(f"   ├── calibration.json")
    print(f"   └── cvat_manifest.json")
    print(f"📦 Ready-to-upload CVAT archive: {zip_export_path} ({zip_export_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
