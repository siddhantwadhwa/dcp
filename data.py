#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
from typing import Tuple, Any, Optional

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class Jitter:
    def __init__(self, sigma=0.01, clip=0.05):
        self._sigma = sigma
        self._clip = clip

    def __call__(self, pointcloud):
        return jitter_pointcloud(pointcloud, self._sigma, self._clip)


class RandomRemove:
    """
    Remove pointcloud randomly with an arbitrary probability.
    """
    def __init__(self, prob=0.2):
        self._prob = prob

    def __call__(self, pointcloud):
        original_length = pointcloud.shape[0]
        random_uniform = np.random.uniform(0, 1, pointcloud.shape[0])
        masked_in = random_uniform > 0.2

        masked_pointcloud = pointcloud[masked_in, :]

        if len(masked_pointcloud) == 0:
            return pointcloud

        while masked_pointcloud.shape[0] < original_length:
            masked_pointcloud = np.concatenate([masked_pointcloud, masked_pointcloud], axis=0)

        masked_pointcloud = masked_pointcloud[:original_length, :]
        return masked_pointcloud


class RandomRemoveFourQuadrant:
    def __call__(self, pointcloud):

        original_length = pointcloud.shape[0]
        along_which_axis = np.random.randint(3)

        boolean_arr_masked_out = np.ones(original_length)
        for i in range(3):
            if i == along_which_axis:
                continue
            is_larger = np.random.randint(2)
            boolean_arr_axis = pointcloud[:, i] > 0 if is_larger else pointcloud[:, i] < 0
            boolean_arr_masked_out = np.logical_and(boolean_arr_axis, boolean_arr_masked_out)

        masked_pointcloud = pointcloud[np.logical_not(boolean_arr_masked_out), :]

        if len(masked_pointcloud) == 0:
            return pointcloud

        while masked_pointcloud.shape[0] < original_length:
            masked_pointcloud = np.concatenate([masked_pointcloud, masked_pointcloud], axis=0)

        masked_pointcloud = masked_pointcloud[:original_length, :]
        return masked_pointcloud


class RandomRemoveTwoQuadrant:
    def __call__(self, pointcloud):
        original_length = pointcloud.shape[0]
        along_which_axis = np.random.randint(3)

        is_larger = np.random.randint(2)
        boolean_mask = pointcloud[:, along_which_axis] > 0 if is_larger else pointcloud[:, along_which_axis] < 0

        masked_pointcloud = pointcloud[boolean_mask, :]

        if len(masked_pointcloud) == 0:
            return pointcloud

        while masked_pointcloud.shape[0] < original_length:
            masked_pointcloud = np.concatenate([masked_pointcloud, masked_pointcloud], axis=0)

        masked_pointcloud = masked_pointcloud[:original_length, :]
        return masked_pointcloud


class RandomRemoveEightQuadrant:
    """
    Remove an eighth of the pointcloud that belongs to a quadrant
    """
    def __call__(self, pointcloud):
        original_length = pointcloud.shape[0]

        larger_x = np.random.randint(2)
        larger_y = np.random.randint(2)
        larger_z = np.random.randint(2)

        boolean_arr_x = pointcloud[:, 0] > 0 if larger_x else pointcloud[:, 0] < 0
        boolean_arr_y = pointcloud[:, 1] > 0 if larger_y else pointcloud[:, 1] < 0
        boolean_arr_z = pointcloud[:, 2] > 0 if larger_z else pointcloud[:, 2] < 0

        masked_out = np.logical_and(boolean_arr_x, boolean_arr_y)
        masked_out = np.logical_and(masked_out, boolean_arr_z)

        masked_pointcloud = pointcloud[np.logical_not(masked_out), :]

        if len(masked_pointcloud) == 0:
            return pointcloud

        while masked_pointcloud.shape[0] < original_length:
            masked_pointcloud = np.concatenate([masked_pointcloud, masked_pointcloud], axis=0)

        masked_pointcloud = masked_pointcloud[:original_length, :]

        return masked_pointcloud


class Augment:
    def __init__(
            self,
            transforms: Tuple[Any] = (
                # Jitter(),  # TODO: ALWAYS LEAVE THIS LINE UNCOMMENTED
                RandomRemoveTwoQuadrant(),  # TODO: Sergi: Uncomment this line and comment out the rest, run python command
                # RandomRemoveFourQuadrant(),  # TODO: Eric: Uncomment this line, and comment out the rest and run the python command.
                # RandomRemoveEightQuadrant(),  # TODO: Siddhant: Uncomment this line and comment out the rest and run the python command.
                # RandomRemove()  # TODO: Arvind: Uncomment this line and comment out the rest and run the python command.
            )
         ):
        self._transforms = transforms
    def __call__(self, pointcloud):
        for transform in self._transforms:
            pointcloud = transform(pointcloud)

        return pointcloud


class ModelNet40(Dataset):
    def __init__(self,
                 num_points,
                 partition='train',
                 gaussian_noise=False,
                 unseen=False,
                 augment: Optional[Augment] = Augment(),
                 is_augment_a: bool = True,  # We are only augmenting the source point cloud throughout the experiments.
                 is_augment_b: bool = False,
                 factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.augment = augment
        self.is_augment_a = is_augment_a
        self.is_augment_b = is_augment_b
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud
        if self.is_augment_a:
            if self.augment:
                pointcloud1 = self.augment(pointcloud)
            else:
                pointcloud1 = pointcloud
        pointcloud1 = pointcloud1.T

        pointcloud2 = pointcloud
        if self.is_augment_b:
            if self.augment:
                pointcloud2 = self.augment(pointcloud)
            else:
                pointcloud2 = pointcloud
        pointcloud2 = pointcloud2.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud2.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
