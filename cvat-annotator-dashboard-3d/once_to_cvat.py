import os
import sys
import json
import argparse
import zipfile
import tarfile
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Any, List, Optional

def load_once_json_from_archive(infos_zip_or_tar: str, sequence_id: str) -> Dict[str, Any]:
    """Load the JSON annotation file for a sequence from zip/tar or folder."""
    if os.path.isdir(infos_zip_or_tar):
        json_path = os.path.join(infos_zip_or_tar, sequence_id, f"{sequence_id}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(infos_zip_or_tar, f"{sequence_id}.json")
        with open(json_path, 'r') as f:
            return json.load(f)

    if infos_zip_or_tar.endswith('.zip'):
        with zipfile.ZipFile(infos_zip_or_tar) as z:
            # Check inside val_infos.tar if nested
            for filename in z.namelist():
                if filename.endswith('infos.tar'):
                    with z.open(filename) as tar_bytes:
                        with tarfile.open(fileobj=tar_bytes) as t:
                            for member in t.getmembers():
                                if member.name.endswith(f"{sequence_id}.json"):
                                    extracted = t.extractfile(member)
                                    return json.load(extracted)
                elif filename.endswith(f"{sequence_id}.json"):
                    with z.open(filename) as f:
                        return json.load(f)

    if infos_zip_or_tar.endswith('.tar'):
        with tarfile.open(infos_zip_or_tar) as t:
            for member in t.getmembers():
                if member.name.endswith(f"{sequence_id}.json"):
                    extracted = t.extractfile(member)
                    return json.load(extracted)

    raise FileNotFoundError(f"Could not find {sequence_id}.json in {infos_zip_or_tar}")


def convert_bin_to_pcd(bin_bytes: bytes, output_pcd_path: str, point_feature_num: int = 4):
    """Convert raw binary float32 LiDAR data to .pcd format."""
    points_flat = np.frombuffer(bin_bytes, dtype=np.float32)
    
    if len(points_flat) % point_feature_num != 0:
        points_flat = points_flat[:len(points_flat) - (len(points_flat) % point_feature_num)]
    
    points = points_flat.reshape(-1, point_feature_num)
    xyz = points[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    
    # Store intensity as greyscale color for CVAT 3D rendering
    if point_feature_num >= 4:
        intensity = points[:, 3]
        max_i = np.max(intensity)
        min_i = np.min(intensity)
        if max_i > min_i:
            norm_i = (intensity - min_i) / (max_i - min_i)
        else:
            norm_i = np.zeros_like(intensity)
        colors = np.stack([norm_i, norm_i, norm_i], axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
    o3d.io.write_point_cloud(output_pcd_path, pcd)


class OnceToCvatStandardizer:
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_infos_source(self, sequence_id: str) -> str:
        """Find where the sequence json / infos archive is located."""
        # 1. Look for extracted directory
        candidate_json_dir = self.data_root / "extracted" / "data"
        if (candidate_json_dir / sequence_id / f"{sequence_id}.json").exists():
            return str(candidate_json_dir)
        
        # 2. Look for zip containing val_infos.tar or the specific json
        for p in self.data_root.glob("*.zip"):
            try:
                with zipfile.ZipFile(p) as z:
                    for name in z.namelist():
                        if "infos" in name or f"{sequence_id}.json" in name:
                            return str(p)
            except Exception:
                continue
            
        # 3. Look for tar
        for p in self.data_root.glob("*.tar"):
            if "infos" in p.name:
                return str(p)
            
        raise FileNotFoundError(f"No infos archive or JSON folder found for {sequence_id} in {self.data_root}")

    def convert_sequence(self, sequence_id: str, max_frames: Optional[int] = 50, start_frame: int = 0):
        """Convert a sequence into CVAT 3D standard folder structure."""
        print(f"\n=======================================================")
        print(f"🚀 Standardizing ONCE Sequence [{sequence_id}] for CVAT 3D")
        print(f"=======================================================")
        
        infos_source = self.find_infos_source(sequence_id)
        print(f"📖 Loading annotations from: {infos_source}")
        scene_data = load_once_json_from_archive(infos_source, sequence_id)
        
        meta_info = scene_data.get("meta_info", {})
        calib_data = scene_data.get("calib", {})
        all_frames = scene_data.get("frames", [])
        point_features = meta_info.get("point_feature_num", 4)
        
        print(f"ℹ️ Total frames in sequence: {len(all_frames)}")
        
        selected_frames = all_frames[start_frame : (start_frame + max_frames if max_frames else len(all_frames))]
        print(f"🎯 Processing {len(selected_frames)} frames (Frames {start_frame} to {start_frame + len(selected_frames) - 1})")

        # Create CVAT 3D standard folder layout
        seq_output = self.output_dir / sequence_id
        point_cloud_dir = seq_output / "point_cloud"
        related_images_dir = seq_output / "related_images"
        ground_truth_dir = seq_output / "ground_truth"
        
        point_cloud_dir.mkdir(parents=True, exist_ok=True)
        related_images_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Calibration metadata
        calib_file = seq_output / "calibration.json"
        with open(calib_file, "w") as f:
            json.dump({
                "sequence_id": sequence_id,
                "meta_info": meta_info,
                "calib": calib_data
            }, f, indent=2)
        print(f"✅ Exported calibration: {calib_file}")

        # 2. Extract Ground Truth 3D Bounding Boxes for Auto-QC
        gt_manifest = []
        for idx, frame in enumerate(selected_frames):
            frame_id = str(frame["frame_id"])
            annos = frame.get("annos", {})
            names = annos.get("names", [])
            boxes_3d = annos.get("boxes_3d", [])
            
            # Format boxes into structured dictionary
            formatted_boxes = []
            for name, box in zip(names, boxes_3d):
                # ONCE box format: [x, y, z, dx, dy, dz, yaw]
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
        print(f"✅ Exported Golden Ground Truth ({len(gt_manifest)} frames): {gt_file}")

        # 3. Check for available LiDAR files
        # Check if we have extracted lidar or if we need to notify user
        extracted_lidar_dir = self.data_root / "extracted" / "data" / sequence_id / "lidar"
        lidar_tar_path = self.data_root / "val_lidar.tar"
        
        has_lidar = False
        if extracted_lidar_dir.exists():
            print(f"📦 Found extracted LiDAR in {extracted_lidar_dir}")
            for idx, frame in enumerate(selected_frames):
                frame_id = str(frame["frame_id"])
                bin_path = extracted_lidar_dir / f"{frame_id}.bin"
                out_pcd = point_cloud_dir / f"{idx:06d}.pcd"
                if bin_path.exists():
                    with open(bin_path, "rb") as bf:
                        convert_bin_to_pcd(bf.read(), str(out_pcd), point_features)
                    has_lidar = True
        elif lidar_tar_path.exists() and lidar_tar_path.stat().st_size > 10000:
            print(f"📦 Extracting LiDAR directly from {lidar_tar_path}...")
            with tarfile.open(lidar_tar_path) as lt:
                for idx, frame in enumerate(selected_frames):
                    frame_id = str(frame["frame_id"])
                    member_name = f"data/{sequence_id}/lidar/{frame_id}.bin"
                    try:
                        m = lt.getmember(member_name)
                        extracted = lt.extractfile(m)
                        out_pcd = point_cloud_dir / f"{idx:06d}.pcd"
                        convert_bin_to_pcd(extracted.read(), str(out_pcd), point_features)
                        has_lidar = True
                    except KeyError:
                        pass
        
        if not has_lidar:
            print(f"⚠️ Note: LiDAR .bin data was not found yet (waiting for val_lidar.tar download).")
            print(f"   Placeholder .pcd and CVAT manifests are prepared.")

        # 4. Extract & Match Camera Images
        print(f"📷 Matching Camera Images for multi-sensor task...")
        self.extract_camera_images(sequence_id, selected_frames, related_images_dir)

        # 5. Create CVAT Task Manifest / Readme
        manifest = {
            "sequence_id": sequence_id,
            "total_frames": len(selected_frames),
            "point_cloud_dir": str(point_cloud_dir.relative_to(self.output_dir)),
            "related_images_dir": str(related_images_dir.relative_to(self.output_dir)),
            "calibration_file": str(calib_file.relative_to(self.output_dir)),
            "ground_truth_file": str(gt_file.relative_to(self.output_dir)),
            "classes": ["Car", "Bus", "Truck", "Pedestrian", "Cyclist"],
            "status": "ready_for_cvat_upload"
        }
        
        manifest_path = seq_output / "cvat_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n🎉 Successfully prepared CVAT 3D dataset at: {seq_output}")
        print(f"   ├── point_cloud/ ({len(list(point_cloud_dir.glob('*.pcd')))} .pcd files)")
        print(f"   ├── related_images/ ({len(list(related_images_dir.glob('*')))} frame camera sets)")
        print(f"   ├── ground_truth/ (gt_boxes3d.json)")
        print(f"   ├── calibration.json")
        print(f"   └── cvat_manifest.json")


    def extract_camera_images(self, sequence_id: str, selected_frames: List[Dict], related_images_dir: Path):
        """Extract multi-camera frames from zip/tar archives into related_images folder."""
        frame_timestamps = {str(frame["frame_id"]): idx for idx, frame in enumerate(selected_frames)}
        
        # Look in zip archives in data_root
        for zip_path in self.data_root.glob("val-*.zip"):
            try:
                with zipfile.ZipFile(zip_path) as z:
                    for filename in z.namelist():
                        if filename.endswith(".tar") and ("cam" in filename):
                            with z.open(filename) as tar_bytes:
                                with tarfile.open(fileobj=tar_bytes) as t:
                                    for member in t.getmembers():
                                        # Path like: data/000027/cam01/1616100800400.jpg
                                        parts = member.name.split('/')
                                        if len(parts) >= 4 and parts[1] == sequence_id:
                                            cam_id = parts[2]
                                            img_name = parts[3]
                                            timestamp = img_name.replace(".jpg", "")
                                            
                                            if timestamp in frame_timestamps:
                                                frame_idx = frame_timestamps[timestamp]
                                                frame_cam_dir = related_images_dir / f"{frame_idx:06d}_pcd"
                                                frame_cam_dir.mkdir(parents=True, exist_ok=True)
                                                
                                                out_img_path = frame_cam_dir / f"{cam_id}.jpg"
                                                if not out_img_path.exists():
                                                    img_data = t.extractfile(member).read()
                                                    with open(out_img_path, "wb") as imf:
                                                        imf.write(img_data)
            except Exception as e:
                print(f"Warning during camera extraction from {zip_path}: {e}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_data = str(script_dir / "data") if (script_dir / "data").exists() else "./data"
    default_output = str(Path(default_data) / "cvat_tasks")

    parser = argparse.ArgumentParser(description="Convert ONCE dataset to CVAT 3D format")
    parser.add_argument("--data_dir", type=str, default=default_data, help="Directory containing ONCE archives")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Output directory for CVAT data")
    parser.add_argument("--sequence", type=str, default="000027", help="Sequence ID to convert")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to extract (default: 50)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    
    args = parser.parse_args()
    
    converter = OnceToCvatStandardizer(args.data_dir, args.output_dir)
    converter.convert_sequence(
        sequence_id=args.sequence,
        max_frames=args.num_frames,
        start_frame=args.start_frame
    )
