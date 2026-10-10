import open3d as o3d
import sys
import os

def visualize_point_cloud(pcd_path):
    if not os.path.exists(pcd_path):
        print(f"Error: Could not find file at {pcd_path}")
        return

    print(f"Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    print(f"Point cloud has {len(pcd.points)} points.")
    
    # Visualize it
    print("Opening 3D viewer. Use your mouse to rotate and zoom.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_pcd.py <path_to_your_file.pcd>")
    else:
        visualize_point_cloud(sys.argv[1])
