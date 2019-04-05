import json
import math
import os
import pickle
from random import randint, uniform

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
from scipy.spatial import distance

from utils import angle_between


def normalize_skeleton(_skel):
    # remove confidence dim
    skel = _skel[:3, :]

    # transpose and scale
    n_joints = skel.shape[1]
    anchor_pt = (skel[:, 3] + skel[:, 9]) / 2.0  # center of shoulders
    skel[:, 0] = anchor_pt  # change neck point
    resize_factor = distance.euclidean(skel[:, 3], skel[:, 9])  # shoulder length
    for i in range(n_joints):
        skel[:, i] = (skel[:, i] - anchor_pt) / resize_factor

    # make it face front
    angle = angle_between(skel[0::2, 3] - skel[0::2, 9], [-1.0, 0.0])
    quaternion = Quaternion(axis=[0, 1, 0], angle=angle)  # radian
    for i in range(n_joints):
        skel[:, i] = quaternion.rotate(skel[:, i])

    return skel


def project_skel(skel_3d):
    skel_2d = skel_3d[:2, :]  # just drop z values

    return skel_2d


def rotate_skel(skel_3d, degree):
    quaternion = Quaternion(axis=[0, 1, 0], angle=math.radians(degree))  # radian
    rotated_skel = np.copy(skel_3d)
    n_joints = skel_3d.shape[1]
    for i in range(n_joints):
        rotated_skel[:, i] = quaternion.rotate(skel_3d[:, i])

    return rotated_skel


def generate_dataset():
    dataset_dir = './panoptic_dataset'

    # traverse directories
    frames = []
    for dir_name, subdir_list, files in os.walk(dataset_dir):
        if 'hdPose3d_stage1' in dir_name:
            print('Found directory: %s' % dir_name)
            for fname in files:
                with open(dir_name + '/' + fname) as f:
                    frames.append(json.load(f))

    # reduce frame rates
    frames = frames[::2]

    # normalize skeletons
    print('normalization...')
    skeletons_3d = []
    for frame in frames:
        for body in frame['bodies']:
            skel = np.array(body['joints15']).reshape((-1, 4)).transpose()
            skel = normalize_skeleton(skel)
            skeletons_3d.append(skel)

    # data augmentation - rotation
    print('augmentation 1...')
    augmented_samples = []
    for skel_3d in skeletons_3d:
        new_sample = rotate_skel(skel_3d, uniform(-20, +20))
        augmented_samples.append(new_sample)

    # data augmentation - random perturbation
    print('augmentation 2...')
    noise_amount = np.std(skeletons_3d[0][:]) / 10.0
    for skel_3d in skeletons_3d:
        noise = np.random.uniform(0, noise_amount, (skel_3d.shape[0], skel_3d.shape[1]))
        new_sample = skel_3d + noise
        new_sample[:, 0] = 0  # neck pos should be zero
        augmented_samples.append(new_sample)
    skeletons_3d.extend(augmented_samples)

    # generate 2D skeletons
    skeletons_2d = []
    for skel_3d in skeletons_3d:
        skel_2d = project_skel(skel_3d)
        skeletons_2d.append(skel_2d)

    # save dataset
    print('saving...')
    dataset = {'3d': skeletons_3d, '2d': skeletons_2d}
    with open('panoptic_dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)


def review_dataset():
    # load
    with open('panoptic_dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)

    # init figures and variables
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4)
    edges_upper = np.array([[1, 2], [1, 4], [4, 5], [5, 6], [1, 10], [10, 11], [11, 12]]) - 1
    colors = ['b', 'darkred', 'r', 'gold', 'darkgreen', 'g', 'lightgreen']

    # draw
    for i in range(2):
        if i == 0:
            ax_3d = ax1
            ax_2d = ax2
        else:
            ax_3d = ax3
            ax_2d = ax4

        # pick a sample
        idx = randint(0, len(dataset['3d'])-1)
        skel_3d = dataset['3d'][idx]
        skel_2d = dataset['2d'][idx]

        # draw 3d
        for edge_i, edge in enumerate(edges_upper):
            ax_3d.plot(skel_3d[0, edge], skel_3d[2, edge], skel_3d[1, edge], color=colors[edge_i])

        ax_3d.set_aspect('equal')
        ax_3d.set_xlabel("x"), ax_3d.set_ylabel("z"), ax_3d.set_zlabel("y")
        ax_3d.set_xlim3d([-2, 2]), ax_3d.set_ylim3d([2, -2]), ax_3d.set_zlim3d([2, -2])
        ax_3d.view_init(elev=10, azim=-45)

        # draw 2d
        for edge_i, edge in enumerate(edges_upper):
            ax_2d.plot(skel_2d[0, edge], skel_2d[1, edge], color=colors[edge_i])

        ax_2d.set_aspect('equal')
        ax_2d.set_xlabel("x"), ax_2d.set_ylabel("y")
        ax_2d.set_xlim([-2, 2]), ax_2d.set_ylim([2, -2])

    plt.show()


if __name__ == "__main__":
    generate_dataset()
    # review_dataset()
