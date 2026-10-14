# 3D Point Cloud Scene Stitcher & Multi-level Smoothing Suite for CVAT

Hệ thống ghép nối mây điểm (Scene Stitching) và làm mượt đa tầng (Multi-level Smoothing) chuyên dụng cho dữ liệu LiDAR xe tự hành (Waymo, nuScenes, KITTI) phục vụ gán nhãn 3D trong CVAT.

---

## 1. Cấu trúc Module (`utils/pointcloud_stitching/`)

- [`pcd_io.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/pcd_io.py): Đọc và ghi file `.pcd` (v0.7) chuẩn, hỗ trợ cả định dạng ASCII và Binary hiệu năng cao.
- [`ego_pose_manager.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/ego_pose_manager.py): Quản lý ma trận biến đổi SE(3) $T_{\text{world} \leftarrow \text{sensor}}$, sắp xếp timeline theo timestamp.
- [`trajectory_smoother.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/trajectory_smoother.py): **Thuật toán làm mượt 1** — B-Spline cho vị trí $(x, y, z)$ và Quaternion SLERP cho góc xoay để triệt tiêu nhiễu rung lắc của GPS/IMU.
- [`icp_registration.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/icp_registration.py): **Thuật toán làm mượt 2** — Fast Rigid ICP dùng `scipy.spatial.cKDTree` và thuật toán Kabsch SVD để vi chỉnh khớp nối giữa các frame kề nhau, **triệt tiêu hoàn toàn lỗi nhân đôi vật thể (Double walls / Ghosting)**.
- [`cloud_smoother.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/cloud_smoother.py):
  - **Thuật toán làm mượt 3** — Voxel Grid Centroid Downsampling: Làm phẳng mặt đường/tường, khống chế số lượng điểm vừa phải để Three.js của CVAT render mượt mà 60 FPS.
  - **Thuật toán làm mượt 4** — Statistical Outlier Removal (SOR): Khử các tia nhiễu bụi, phản xạ ảo của cảm biến.
- [`scene_stitcher.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/scene_stitcher.py): Bộ điều phối trung tâm (Pipeline orchestrator) kết hợp tất cả các bước trên.
- [`cli.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/cli.py): Giao diện dòng lệnh (CLI).
- [`test_stitching.py`](file:///media/bang/Work/projects/cvat/utils/pointcloud_stitching/test_stitching.py): Bộ kiểm thử tự động (Unit & Integration tests).

---

## 2. Hướng dẫn sử dụng CLI

Chạy trực tiếp từ thư mục gốc của repository:

```bash
# Ghép các khung hình với cửa sổ trượt +/- 2 frames và kích thước voxel 5cm
python3 utils/pointcloud_stitching/cli.py \
    --poses waymo_sample/extracted/ego_poses.json \
    --pcd-dir waymo_sample/extracted/lidar/pcd \
    --output-dir waymo_sample/extracted/lidar/stitched_pcd \
    --window 2 \
    --voxel-size 0.05
```

### Các tùy chọn tham số:
- `--poses`: Đường dẫn tới file `ego_poses.json`.
- `--pcd-dir`: Thư mục chứa các file `.pcd` gốc.
- `--output-dir`: Thư mục lưu các file `.pcd` đã ghép và làm mượt.
- `--window`: Bán kính cửa sổ trượt (mặc định: `2`, tức là ghép $t-2, t-1, t, t+1, t+2$).
- `--voxel-size`: Kích thước voxel (mét) để làm phẳng bề mặt (mặc định: `0.05`m = 5cm).
- `--no-traj-smooth`: Tắt làm mượt quỹ đạo B-Spline/SLERP.
- `--no-icp`: Tắt vi chỉnh ICP.
- `--no-sor`: Tắt bộ lọc nhiễu SOR.
- `--binary`: Lưu file PCD dưới dạng binary (nhẹ hơn và nạp nhanh hơn).

---

## 3. Sử dụng trực tiếp trong Python Code

```python
from utils.pointcloud_stitching import EgoPoseManager, SceneStitcher

# 1. Nạp Ego-poses
mgr = EgoPoseManager("waymo_sample/extracted/ego_poses.json")

# 2. Khởi tạo Stitcher với các thuật toán làm mượt
stitcher = SceneStitcher(
    ego_pose_mgr=mgr,
    window_size=2,
    voxel_size=0.04,
    enable_trajectory_smoothing=True,
    enable_icp=True,
    enable_sor=True
)

# 3. Ghép và làm mượt toàn bộ sequence
results = stitcher.process_sequence(
    pcd_files_map=pcd_map,
    output_dir="output_stitched_pcd/",
    binary=False
)
```

---

## 4. Chạy kiểm thử tự động (Test Suite)

```bash
python3 utils/pointcloud_stitching/test_stitching.py
```
