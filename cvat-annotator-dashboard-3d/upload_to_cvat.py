import json
import time
import requests

CVAT_URL = "http://localhost:8080"
AUTH_TOKEN = "e969c9a02bb584f5b1fb0371a9c77031fa52aaa0"
HEADERS = {"Authorization": f"Token {AUTH_TOKEN}"}

def main():
    print("[*] 1. Creating 3D Task in CVAT...")
    task_payload = {
        "name": "ONCE_000201_39frames_3D_Tracking",
        "labels": [
            {"name": "Car", "color": "#3366ff"},
            {"name": "Bus", "color": "#ff9900"},
            {"name": "Truck", "color": "#ff3300"},
            {"name": "Pedestrian", "color": "#33cc33"},
            {"name": "Cyclist", "color": "#9933ff"}
        ],
        "dimension": "3d",
        "segment_size": 39
    }
    
    resp = requests.post(f"{CVAT_URL}/api/tasks", headers=HEADERS, json=task_payload)
    if resp.status_code != 201:
        print(f"[-] Failed to create task: {resp.status_code} {resp.text}")
        return
        
    task_data = resp.json()
    task_id = task_data["id"]
    print(f"[+] Task created successfully! Task ID: {task_id}")
    
    # 2. Upload Data
    print(f"[*] 2. Uploading 3D dataset zip to Task #{task_id}...")
    zip_path = "/tmp/000201_39frames_cvat.zip"
    
    with open(zip_path, "rb") as zf:
        files = {"client_files[0]": ("000201_39frames_cvat.zip", zf, "application/zip")}
        data_payload = {
            "image_quality": 70,
            "use_zip_chunks": True,
            "use_cache": True
        }
        resp = requests.post(
            f"{CVAT_URL}/api/tasks/{task_id}/data",
            headers=HEADERS,
            data=data_payload,
            files=files
        )
        
    if resp.status_code not in [200, 201, 202]:
        print(f"[-] Data upload failed: {resp.status_code} {resp.text}")
        return
        
    print(f"[+] Data upload initiated: {resp.status_code}")
    
    # 3. Poll for Task completion
    print("[*] 3. Waiting for CVAT chunk worker to process point clouds & images...")
    max_wait = 90
    start_time = time.time()
    task_ready = False
    
    while time.time() - start_time < max_wait:
        resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/status", headers=HEADERS)
        if resp.status_code == 200:
            status_data = resp.json()
            state = status_data.get("state")
            message = status_data.get("message", "")
            print(f"    Status: {state} ({message})")
            if state == "Completed":
                task_ready = True
                break
            elif state == "Failed":
                print(f"[-] Task data processing failed: {message}")
                return
        time.sleep(2)
        
    if not task_ready:
        print("[-] Timed out waiting for task to process.")
        return
        
    print(f"[+] Task #{task_id} data is fully processed!")
    
    # 4. Get Task Jobs and Labels
    resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}", headers=HEADERS)
    task_details = resp.json()
    jobs = task_details.get("jobs", [])
    if not jobs:
        resp = requests.get(f"{CVAT_URL}/api/jobs?task_id={task_id}", headers=HEADERS)
        jobs = resp.json().get("results", [])
        
    if not jobs:
        print("[-] No jobs found for task.")
        return
        
    job_id = jobs[0]["id"]
    print(f"[+] Associated Job ID: {job_id}")
    
    # Map labels
    label_map = {lbl["name"]: lbl["id"] for lbl in task_details.get("labels", [])}
    print(f"[+] Label Map: {label_map}")
    
    # 5. Import 3D Bounding Boxes as Annotations
    print(f"[*] 4. Uploading 3D Cuboids into Job #{job_id}...")
    with open("/tmp/gt_boxes3d.json", "r") as gf:
        gt_manifest = json.load(gf)
        
    shapes = []
    for frame_idx, frame_data in enumerate(gt_manifest):
        boxes = frame_data.get("boxes_3d", [])
        for b in boxes:
            label_name = b.get("label", "Car")
            lbl_id = label_map.get(label_name)
            if not lbl_id:
                continue
                
            center = b["center"]
            size = b["size"]
            yaw = b.get("rotation_yaw", 0.0)
            
            # CVAT 3D cuboid 16-element points array:
            # [x, y, z, rx, ry, rz, dx, dy, dz, 0, 0, 0, 0, 0, 0, 0]
            points_16 = [
                float(center[0]), float(center[1]), float(center[2]),
                0.0, 0.0, float(yaw),
                float(size[0]), float(size[1]), float(size[2]),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
            
            shapes.append({
                "type": "cuboid",
                "frame": frame_idx,
                "label_id": lbl_id,
                "points": points_16,
                "occluded": False
            })
            
    print(f"[*] Total 3D cuboid annotations to insert: {len(shapes)}")
    
    # Send in chunks if large
    chunk_size = 500
    for chunk_start in range(0, len(shapes), chunk_size):
        chunk = shapes[chunk_start : chunk_start + chunk_size]
        anno_payload = {
            "shapes": chunk,
            "tracks": [],
            "tags": []
        }
        anno_resp = requests.patch(
            f"{CVAT_URL}/api/jobs/{job_id}/annotations?action=create",
            headers=HEADERS,
            json=anno_payload
        )
        if anno_resp.status_code not in [200, 201]:
            print(f"[-] Annotation upload failed: {anno_resp.status_code} {anno_resp.text}")
        else:
            print(f"    ✓ Inserted batch {chunk_start + len(chunk)}/{len(shapes)}")
            
    print("\n🎉 SUCCESS! Dataset & Annotations successfully pushed to CVAT!")
    print(f"🔗 Direct Task URL: {CVAT_URL}/tasks/{task_id}")
    print(f"🔗 Direct Job URL:  {CVAT_URL}/tasks/{task_id}/jobs/{job_id}")

if __name__ == "__main__":
    main()
