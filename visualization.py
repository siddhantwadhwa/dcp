# Utils to visualize pointclouds

import numpy as np
import open3d as o3d

from o3d_utils import convert_np_array_to_o3d_pcd

def generate_random_color():
    """Generates a randomly generate list with 3 elements b/w 0 and 1.

    Returns:
        list: containing 3 floating point values b/w 0 and 1. 
    """
    return list(np.random.choice(range(100),size=3) / 100)

def visualize_pointclouds(pcds, colors=None):
    """Displays multiple pointclouds in the same open3d canvas.

    Args:
        pcds: Either (1) list of numpy arrays of shape (N,3) each row
              representing a point in xyz format, OR (2) list of
              open3d.geometry.PointCloud objects.
        colors (optional): list of lists of size 3. Defaults to None.
    """
    assert len(pcds) > 0, "No pointcloud provided."
    if not isinstance(pcds, list):
        pcds = [pcds]

    # Populate colors for each pointcloud, if not provided
    if colors is None:
        colors = []
        for _ in range(len(pcds)):
            colors.append(generate_random_color())
    
    # Check that all list elements have the same type
    for elem in pcds:
        print(type(elem))
    assert len(set([type(elem) for elem in pcds])) == 1

    if isinstance(pcds[0], np.ndarray):
        visualize_nparray_pointclouds(pcds, colors)
    elif isinstance(pcds[0], o3d.geometry.PointCloud):
        visualize_o3d_pointclouds(pcds, colors)
    else:
        raise TypeError


def visualize_nparray_pointclouds(pc_arrays, colors):
    """Displays multiple pointclouds in the same open3d canvas.

    Args:
        pc_arrays: list of numpy arrays of shape (N,3) each row 
                            representing a point in xyz format.
        colors: list of lists of size 3. Defaults to None.
    """
    if not isinstance(pc_arrays, list):
        pc_arrays = [pc_arrays]
    # Check inputs
    assert len(pc_arrays) > 0, "No pointcloud provided."
    for pc_array in pc_arrays:
        assert pc_array.shape[1] == 3, "Represent each point as numpy row: [x y z]"
    for color in colors:
        assert len(color) == 3
    assert len(colors) == len(pc_arrays)
    
    pcds_to_display = []
    for idx, pc_array in enumerate(pc_arrays):
        pcd = convert_np_array_to_o3d_pcd(pc_array)
        pcd.paint_uniform_color(colors[idx])
        pcds_to_display.append(pcd)

    # Draw Canvas
    o3d.visualization.draw_geometries(pcds_to_display)


def visualize_o3d_pointclouds(o3d_pcds, colors):
    """Displays multiple pointclouds in the same open3d canvas.

    Args:
        o3d_pcds: list of open3d.geometry.PointCloud objects.
        colors: list of lists of size 3. Defaults to None.
    """
    if not isinstance(o3d_pcds, list):
        o3d_pcds = [o3d_pcds]
    # Check inputs
    assert len(o3d_pcds) > 0, "No pointcloud provided."
    for color in colors:
        assert len(color) == 3
    assert len(colors) == len(o3d_pcds)
    
    pcds_to_display = []
    for idx, pcd in enumerate(o3d_pcds):
        pcd.paint_uniform_color(colors[idx])
        pcds_to_display.append(pcd)

    # Draw Canvas
    o3d.visualization.draw_geometries(pcds_to_display)