#!/usr/bin/env python

# Tool to visualize a list of ply files on the same canvas.

from absl import app
from absl import flags
import numpy as np
import sys

from file_io import read_ply_file_as_o3d_pcd
from visualization import visualize_pointclouds

FLAGS = flags.FLAGS
flags.DEFINE_list('files', None, 'Comma separated list of ply files to visualize')
flags.DEFINE_integer('num_points', 2048, 'Number of lidar points to load for each model')


def main(argv):
    pcds = []
    for file_path in FLAGS.files:
        pcds.append(read_ply_file_as_o3d_pcd(file_path))
    visualize_pointclouds(pcds)

if __name__ == '__main__':
    app.run(main)