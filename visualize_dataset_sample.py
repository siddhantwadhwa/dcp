#!/usr/bin/env python

# Tool to visualize a dataset sample

from absl import app
from absl import flags
import numpy as np
import sys

from data import ModelNet40
from visualization import visualize_pointclouds

FLAGS = flags.FLAGS
flags.DEFINE_integer('index', 0, 'index of sample in dataset')
flags.DEFINE_integer('num_points', 2048, 'Number of lidar points to load for each model')


def main(argv):
    dataset = ModelNet40(num_points=FLAGS.num_points)
    pointcloud1, pointcloud2, R_ab, translation_ab, _, _, euler_ab, _ = dataset[FLAGS.index]
    visualize_pointclouds([pointcloud1.T, pointcloud2.T])

if __name__ == '__main__':
    app.run(main)