import pandas as pd
import open3d as o3d
import numpy as np
import sys
import os

def convert_pandaset_pkl_to_pcd(pkl_path, output_pcd_path):
    print(f"Loading {pkl_path}...")
    
    # PandaSet LiDAR data is stored as a pickled Pandas DataFrame
    # Columns usually include: x, y, z, i (intensity), d (device ID), t (timestamp)
    df = pd.read_pickle(pkl_path)
    
    # Extract the XYZ coordinates
    xyz = df[['x', 'y', 'z']].to_numpy()
    
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Optional: If you want to include intensity as colors (CVAT can display this)
    if 'i' in df.columns:
        # Normalize intensity to 0.0 - 1.0 for coloring
        intensity = df['i'].to_numpy()
        intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-6)
        # Create a greyscale color array based on intensity
        colors = np.stack((intensity_norm, intensity_norm, intensity_norm), axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to standard .pcd format
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"✅ Successfully saved to {output_pcd_path} ({len(xyz)} points)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_pkl_to_pcd.py input.pkl output.pcd")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)
        
    convert_pandaset_pkl_to_pcd(input_file, output_file)
