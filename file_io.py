# Utils for file IO

import numpy as np
import open3d as o3d

def write_o3d_pcd_to_ply_file(o3d_pcd, file_path):
    """Writes open3d.geometry.PointCloud object to ply file."""
    assert isinstance(o3d_pcd, o3d.geometry.PointCloud)
    assert file_path.endswith(".ply")
    o3d.io.write_point_cloud(file_path, o3d_pcd)

def write_array_to_ply_file(pc_array, file_path):
    """Writes pointcloud in np.ndarray format to ply file."""
    assert isinstance(pc_array, np.ndarray)
    assert file_path.endswith(".ply")
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pc_array)
    write_o3d_pcd_to_ply_file(o3d_pcd, file_path)

def read_ply_file_as_o3d_pcd(file_path):
    """Reads .pply file and returns open3d.geometry.PointCloud object."""
    return o3d.io.read_point_cloud(file_path)

def read_ply_file_as_np_array(file_path):
    """Reads .pply file and returns pointcloud in np.ndarray format."""
    o3d_pcd = read_ply_file_as_o3d_pcd(file_path)
    assert o3d_pcd is not None
    return np.asarray(o3d_pcd.points)