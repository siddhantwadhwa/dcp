# Open3D utils

import numpy as np
import open3d as o3d
import tempfile

from file_io import read_ply_file_as_o3d_pcd, write_array_to_ply_file

def convert_np_array_to_o3d_pcd(pcd_array):
    """Converts an np array containing pointcloud data to an Open3D Pointcloud type.

    Args:
        pcd_array (np.ndarray): ndarray with shape: (N,3) containing points in xyz format.

    Returns:
        open3d.geometry.PointCloud: Open3D pointcloud structure containing points from pcd_array.
    """
    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        # Need to write to file, then re-read to convert
        # array into o3d poinctloud object
        # http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html
        write_array_to_ply_file(pcd_array, f.name)
        pcd = read_ply_file_as_o3d_pcd(f.name)
    return pcd