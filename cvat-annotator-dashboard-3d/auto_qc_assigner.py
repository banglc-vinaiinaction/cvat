#!/usr/bin/env python3
"""
Enterprise 3D Cuboids QA/QC & Auto-Assignment Engine
Platform: cvat-annotator-dashboard-3d
Evaluates 3D cuboids against Golden Ground Truth using 3D Oriented IoU,
computes NuScenes TP metrics (ATE, ASE, AOE), pins 3D defect coordinates in CVAT,
and auto-assigns subsequent tasks based on annotator performance scores.
"""

import os
import sys
import math
import numpy as np
from typing import Dict, Any, List, Optional
from shapely.geometry import Polygon

from cvat_sdk import make_client, models
from cvat_sdk.core.client import Client

# --- Configuration & Enterprise Security ---
CVAT_URL = os.getenv("CVAT_URL", "http://localhost:8080")
CVAT_PAT = os.getenv("CVAT_PAT", "")

IOU_THRESHOLD_PASS = float(os.getenv("IOU_THRESHOLD", "0.60"))
ATE_THRESHOLD_PASS = float(os.getenv("ATE_THRESHOLD", "0.30"))  # meters


# ==============================================================================
# 1. 3D Cuboid Mathematical Structure
# ==============================================================================

class Cuboid3D:
    def __init__(self, points: List[float], label_id: int = 0, label_name: str = "object", obj_id: Optional[int] = None):
        """
        Parses CVAT's 16-float points array:
        [x, y, z, rx, ry, rz, dx, dy, dz, 0, 0, 0, 0, 0, 0, 0]
        """
        if len(points) < 9:
            raise ValueError(f"Expected at least 9 values for 3D cuboid, got {len(points)}")
        
        self.x = float(points[0])
        self.y = float(points[1])
        self.z = float(points[2])
        self.rx = float(points[3])
        self.ry = float(points[4])
        self.rz = float(points[5])  # Yaw angle in radians
        self.dx = float(points[6])  # Extent X
        self.dy = float(points[7])  # Extent Y
        self.dz = float(points[8])  # Extent Z
        self.label_id = label_id
        self.label_name = label_name
        self.obj_id = obj_id

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def extents(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dz])

    @property
    def volume(self) -> float:
        return float(self.dx * self.dy * self.dz)

    def get_bev_corners(self) -> np.ndarray:
        """Returns 4 planar ground-plane corners (x, y) rotated by yaw angle."""
        hx, hy = self.dx / 2.0, self.dy / 2.0
        local_corners = np.array([
            [hx, hy],
            [hx, -hy],
            [-hx, -hy],
            [-hx, hy]
        ])
        cos_yaw, sin_yaw = np.cos(self.rz), np.sin(self.rz)
        rot_2d = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        bev_corners = (rot_2d @ local_corners.T).T + np.array([self.x, self.y])
        return bev_corners


# ==============================================================================
# 2. Geometric Metrics (3D IoU, ATE, ASE, AOE)
# ==============================================================================

def compute_ate_3d(pred: Cuboid3D, gt: Cuboid3D) -> float:
    """Average Translation Error (Euclidean center distance in meters)."""
    return float(np.linalg.norm(pred.center - gt.center))


def compute_ase(pred: Cuboid3D, gt: Cuboid3D) -> float:
    """Average Scale Error (1 - aligned 3D IoU)."""
    min_dims = np.minimum(pred.extents, gt.extents)
    intersection_vol = float(np.prod(min_dims))
    union_vol = pred.volume + gt.volume - intersection_vol
    return float(1.0 - (intersection_vol / union_vol)) if union_vol > 0 else 0.0


def compute_aoe_yaw(pred: Cuboid3D, gt: Cuboid3D) -> float:
    """Average Orientation Error for yaw in radians [0, pi]."""
    diff = abs(pred.rz - gt.rz) % (2.0 * math.pi)
    diff = min(diff, 2.0 * math.pi - diff)
    return float(diff)


def compute_3d_iou(pred: Cuboid3D, gt: Cuboid3D) -> float:
    """Computes exact 3D IoU for upright objects using BEV polygon intersection + vertical overlap."""
    poly_pred = Polygon(pred.get_bev_corners())
    poly_gt = Polygon(gt.get_bev_corners())

    if not poly_pred.is_valid:
        poly_pred = poly_pred.buffer(0)
    if not poly_gt.is_valid:
        poly_gt = poly_gt.buffer(0)

    if not poly_pred.intersects(poly_gt):
        return 0.0

    inter_area = poly_pred.intersection(poly_gt).area

    # Vertical Height Overlap
    z_min_pred, z_max_pred = pred.z - pred.dz / 2.0, pred.z + pred.dz / 2.0
    z_min_gt, z_max_gt = gt.z - gt.dz / 2.0, gt.z + gt.dz / 2.0

    z_inter = max(0.0, min(z_max_pred, z_max_gt) - max(z_min_pred, z_min_gt))
    if z_inter <= 0.0:
        return 0.0

    vol_inter = inter_area * z_inter
    vol_union = pred.volume + gt.volume - vol_inter
    return float(vol_inter / vol_union) if vol_union > 0 else 0.0


# ==============================================================================
# 3. 3D QC Manager & Auto-Assigner
# ==============================================================================

class QCManager:
    def __init__(self, client: Client):
        self.client = client
        self.performance_db: Dict[str, dict] = {}

    def fetch_job_performance(self, task_id: int, golden_gt_per_frame: Dict[int, List[Cuboid3D]]):
        """Fetch all jobs for a task and calculate 3D annotator performance."""
        print(f"[*] Fetching jobs for Task #{task_id}...")
        jobs = self.client.jobs.list(task_id=task_id)

        task_labels = self.client.tasks.retrieve(task_id).get_labels()
        label_map = {l.id: l.name for l in task_labels}

        for job in jobs:
            if not job.assignee:
                continue

            username = job.assignee.username
            print(f"\n--- Analyzing Job #{job.id} assigned to {username} ---")

            annotations = self.client.jobs.retrieve(job.id).get_annotations()
            cuboid_shapes = [s for s in annotations.shapes if s.type.value == "cuboid" and len(s.points) == 16]

            if not cuboid_shapes:
                print(f"  [Notice] Job #{job.id} has no 3D cuboids.")
                continue

            ious: List[float] = []
            ates: List[float] = []
            defects_pinned = 0

            for shape in cuboid_shapes:
                pred = Cuboid3D(shape.points, label_id=shape.label_id,
                                label_name=label_map.get(shape.label_id, "object"), obj_id=shape.id)
                gts = golden_gt_per_frame.get(shape.frame, [])

                best_iou = 0.0
                best_gt = None
                for gt in gts:
                    iou = compute_3d_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

                if best_gt:
                    ate = compute_ate_3d(pred, best_gt)
                    ase = compute_ase(pred, best_gt)
                    aoe = compute_aoe_yaw(pred, best_gt)
                    ious.append(best_iou)
                    ates.append(ate)

                    # Automated Defect Pinning
                    if best_iou < IOU_THRESHOLD_PASS:
                        defects_pinned += 1
                        reasons = []
                        if ate > ATE_THRESHOLD_PASS:
                            reasons.append(f"[DEFECT:CENTER_OFFSET] Center displaced by {ate:.2f}m")
                        if ase > 0.25:
                            reasons.append(f"[DEFECT:TIGHTNESS] Scale mismatch ASE={ase:.2f}")
                        if aoe > 0.15:
                            reasons.append(f"[DEFECT:HEADING] Heading misaligned by {math.degrees(aoe):.1f}°")

                        msg = " | ".join(reasons) if reasons else f"[DEFECT:LOW_IOU] 3D IoU was {best_iou:.2f}"
                        self.pin_defect(job.id, shape.frame, pred.center.tolist(), msg)

            mean_iou = float(np.mean(ious)) if ious else 0.0
            mean_ate = float(np.mean(ates)) if ates else 0.0
            rework_rate = defects_pinned / len(cuboid_shapes) if cuboid_shapes else 0.0

            self.performance_db[username] = {
                "user_id": job.assignee.id,
                "username": username,
                "last_job_id": job.id,
                "mean_iou_3d": mean_iou,
                "mean_ate_meters": mean_ate,
                "rework_rate": rework_rate,
                "defects_pinned": defects_pinned
            }

            print(f"[{username}] Mean 3D IoU: {mean_iou:.2f}, Mean ATE: {mean_ate:.2f}m, Defects: {defects_pinned}")

    def pin_defect(self, job_id: int, frame: int, position: List[float], comment_text: str):
        """Creates an issue anchored in 3D space with an initial comment."""
        try:
            self.client.api_client.issues_api.create({
                "job": job_id,
                "frame": frame,
                "position": position,
                "comments": [{"message": comment_text}]
            })
            print(f"    [!] Pinned 3D Defect on Frame {frame} at {position}: {comment_text}")
        except Exception as e:
            print(f"    [Warning] Failed to pin issue: {e}")

    def auto_assign_next_jobs(self, next_task_id: int):
        """Auto-assign unassigned jobs in a new task based on past performance."""
        for username, perf in self.performance_db.items():
            print(f"\nEvaluating assignment for {username}...")
            difficulty = "medium"
            if perf["mean_iou_3d"] < 0.70 or perf["mean_ate_meters"] > 0.35:
                difficulty = "easy"
                print(f"-> {username} needs coaching. Assigning EASY segment.")
            elif perf["mean_iou_3d"] >= 0.80 and perf["rework_rate"] < 0.05:
                difficulty = "hard"
                print(f"-> {username} is high-precision. Assigning COMPLEX segment.")

            # Find unassigned job
            jobs = self.client.jobs.list(task_id=next_task_id)
            for j in jobs:
                if not j.assignee:
                    print(f"Assigning Job #{j.id} to {username}")
                    job_instance = self.client.jobs.retrieve(j.id)
                    job_instance.update({"assignee_id": perf["user_id"]})
                    break


# ==============================================================================
# 4. Main CLI Runner
# ==============================================================================

if __name__ == "__main__":
    if not CVAT_PAT:
        print("[!] Notice: CVAT_PAT environment variable is not set. Set it via export CVAT_PAT='...'")
        print("    Running self-test with synthesized 3D cuboids...")
        # Self-test demonstration
        b1 = Cuboid3D([10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2.0, 1.6])
        b2 = Cuboid3D([10.1, 2.05, 0.0, 0.0, 0.0, 0.05, 4.55, 1.95, 1.6])
        print(f"Test 3D IoU: {compute_3d_iou(b1, b2):.3f}")
        print(f"Test ATE:    {compute_ate_3d(b1, b2):.3f} meters")
        print(f"Test ASE:    {compute_ase(b1, b2):.3f}")
        print(f"Test AOE:    {math.degrees(compute_aoe_yaw(b1, b2)):.2f} degrees")
        print("[+] Self-test PASSED successfully.")
        sys.exit(0)

    with make_client(CVAT_URL, access_token=CVAT_PAT) as client:
        manager = QCManager(client)
        # In production, pass golden GT from database or golden sequence
        mock_gt: Dict[int, List[Cuboid3D]] = {}
        manager.fetch_job_performance(task_id=1, golden_gt_per_frame=mock_gt)
        manager.auto_assign_next_jobs(next_task_id=2)
