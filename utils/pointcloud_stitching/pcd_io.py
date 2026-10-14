# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
PCD (Point Cloud Data) Reader and Writer
Supports both ASCII and Binary formats for CVAT 3D point cloud workflows.
Compatible with Open3D, CloudCompare, PCL, and CVAT Three.js PCDLoader.
"""

import os
import struct
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

def read_pcd(filepath: str) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Reads a PCD file into a numpy array and metadata dictionary.

    Returns:
        points: (N, D) float32 numpy array
        fields: list of field names (e.g. ['x', 'y', 'z', 'intensity'])
        metadata: dictionary of PCD header keys
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PCD file not found: {filepath}")

    metadata: Dict[str, Any] = {}
    header_lines = []

    with open(filepath, 'rb') as f:
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                break
            line = line_bytes.decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            val = parts[1:]
            metadata[key] = val

            if key == 'DATA':
                data_format = val[0].lower()
                break

        fields = metadata.get('FIELDS', ['x', 'y', 'z'])
        sizes = [int(s) for s in metadata.get('SIZE', [4] * len(fields))]
        types = metadata.get('TYPE', ['F'] * len(fields))
        counts = [int(c) for c in metadata.get('COUNT', [1] * len(fields))]
        num_points = int(metadata.get('POINTS', [metadata.get('WIDTH', [0])[0]])[0])

        total_cols = sum(counts) if counts else len(fields)

        if data_format == 'ascii':
            raw_text = f.read().decode('ascii', errors='ignore')
            flat = np.fromstring(raw_text, dtype=np.float64, sep=' ')
            if flat.size == 0:
                return np.empty((0, total_cols), dtype=np.float32), fields, metadata
            mat_f64 = flat.reshape(-1, total_cols)
            data = mat_f64.astype(np.float32)
            col_idx = 0
            for name, t, count in zip(fields, types, counts):
                if count == 1 and name.lower() in ('rgb', 'rgba') and t.upper() in ('U', 'I'):
                    u32 = np.rint(mat_f64[:, col_idx]).astype(np.uint32)
                    data[:, col_idx] = u32.view(np.float32)
                col_idx += count
            return data, fields, metadata
        elif data_format == 'binary':
            raw_bytes = f.read()
            if num_points == 0 or not raw_bytes:
                return np.empty((0, total_cols), dtype=np.float32), fields, metadata

            # Fast path when all fields are standard 4-byte fields (float32 or packed 4-byte rgb/rgba)
            if all(
                s == 4 and (t.upper() == 'F' or name.lower() in ('rgb', 'rgba'))
                for name, t, s in zip(fields, types, sizes)
            ):
                expected_floats = num_points * total_cols
                data = np.frombuffer(
                    raw_bytes, dtype=np.float32, count=expected_floats
                ).reshape(num_points, total_cols).copy()
                return data, fields, metadata

            # Create numpy dtype from header for heterogeneous field types
            type_map = {'F': 'f', 'U': 'u', 'I': 'i'}
            dtype_fields = []
            for name, size, t, count in zip(fields, sizes, types, counts):
                np_code = type_map.get(t.upper(), 'f') + str(size)
                if count > 1:
                    dtype_fields.append((name, np_code, (count,)))
                else:
                    dtype_fields.append((name, np_code))

            dt = np.dtype(dtype_fields)
            structured_data = np.frombuffer(raw_bytes, dtype=dt, count=num_points)

            # Convert structured array into preallocated float32 array
            data = np.empty((num_points, total_cols), dtype=np.float32)
            col_idx = 0
            for name, size, t, count in zip(fields, sizes, types, counts):
                col = structured_data[name]
                if count > 1:
                    data[:, col_idx:col_idx + count] = col
                    col_idx += count
                else:
                    if name.lower() in ('rgb', 'rgba') and size == 4 and t.upper() in ('U', 'I'):
                        data[:, col_idx] = np.ascontiguousarray(col, dtype=np.uint32).view(np.float32)
                    else:
                        data[:, col_idx] = col
                    col_idx += 1
            return data, fields, metadata
        else:
            raise ValueError(f"Unsupported PCD DATA format: {data_format}")

def write_pcd(
    filepath: str,
    points: np.ndarray,
    fields: Optional[List[str]] = None,
    binary: bool = False
) -> None:
    """
    Writes points to a PCD file (v0.7).
    
    Args:
        filepath: destination path
        points: (N, D) numpy array
        fields: names of columns, default ['x', 'y', 'z', 'intensity'] if D>=4 else ['x', 'y', 'z']
        binary: if True, writes in binary format for fast loading and smaller size
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_points, num_dims = points.shape

    if fields is None:
        default_fields = ['x', 'y', 'z', 'intensity', 'ring', 'timestamp']
        fields = default_fields[:num_dims]
        while len(fields) < num_dims:
            fields.append(f'f_{len(fields)}')

    sizes = [4] * num_dims
    types = [
        ('U' if (not binary and f_name.lower() in ('rgb', 'rgba')) else 'F')
        for f_name in fields
    ]
    counts = [1] * num_dims

    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(s) for s in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(str(c) for c in counts)}\n"
        f"WIDTH {num_points}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_points}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )

    with open(filepath, 'wb' if binary else 'w') as f:
        if binary:
            f.write(header.encode('ascii'))
            points_f32 = np.ascontiguousarray(points, dtype=np.float32)
            f.write(points_f32.tobytes())
        else:
            f.write(header)
            rgb_cols = [idx for idx, f_name in enumerate(fields) if f_name.lower() in ('rgb', 'rgba')]
            if not rgb_cols:
                np.savetxt(f, points, fmt='%.4f')
            else:
                out_arr = np.empty((num_points, num_dims), dtype=np.float64)
                fmt_list = []
                for idx in range(num_dims):
                    if idx in rgb_cols:
                        u32 = np.ascontiguousarray(points[:, idx], dtype=np.float32).view(np.uint32)
                        out_arr[:, idx] = u32.astype(np.float64)
                        fmt_list.append('%d')
                    else:
                        out_arr[:, idx] = points[:, idx]
                        fmt_list.append('%.4f')
                np.savetxt(f, out_arr, fmt=' '.join(fmt_list))
