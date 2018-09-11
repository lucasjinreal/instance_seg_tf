import os
from glob import glob
import numpy as np
import scipy.misc
import random
from sklearn.utils import shuffle
import shutil
import time
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import glog as log

# Mean and std deviation for whole training data set (RGB format)
mean = np.array([92.14031982, 103.20146942, 103.47182465])
std = np.array([49.157, 54.9057, 59.4065])

INSTANCE_COLORS = [np.array([0, 0, 0]),
                   np.array([20., 20., 20.]),
                   np.array([70., 70., 70.]),
                   np.array([120., 120., 120.]),
                   np.array([170., 170., 170.]),
                   np.array([220., 220., 220.])
                   ]


def get_batches_fn(batch_size, image_shape, image_paths, label_paths):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """

    # print ('Number of total labels:', len(label_paths))
    assert len(image_paths) == len(label_paths), 'Number of images and labels do not match'

    image_paths.sort()
    label_paths.sort()

    # image_paths = image_paths[:10]
    # label_paths = label_paths[:10]

    image_paths, label_paths = shuffle(image_paths, label_paths)
    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        for image_file, gt_image_file in zip(image_paths[batch_i:batch_i + batch_size],
                                             label_paths[batch_i:batch_i + batch_size]):
            image = cv2.resize(cv2.imread(image_file), image_shape, interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = (image.astype(np.float32)-mean)/std

            gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)
            gt_image = cv2.resize(gt_image[:, :, 0], image_shape, interpolation=cv2.INTER_NEAREST)

            images.append(image)
            gt_images.append(gt_image)
        yield np.array(images), np.array(gt_images)


def get_validation_batch_lane(data_dir, image_shape):
    valid_image_paths = glob(os.path.join(data_dir, 'images', '*.jpg'))
    valid_label_paths = glob(os.path.join(data_dir, 'masks', '*.png'))

    #
    # valid_image_paths = [os.path.join(data_dir,'images','0000.png')]
    # valid_label_paths = [os.path.join(data_dir,'labels','0000.png')]

    images = []
    gt_images = []
    for image_file, gt_image_file in zip(valid_image_paths, valid_label_paths):
        image = cv2.resize(cv2.imread(image_file), image_shape, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = (image.astype(np.float32)-mean)/std

        gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)
        gt_image = cv2.resize(gt_image[:, :, 0], image_shape, interpolation=cv2.INTER_NEAREST)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)


def get_validation_batch_cityscapes(image_shape, batch_size=10):
    images, labels = get_cityscapes_f_paths('/media/jintian/sg/permanent/datasets/Cityscapes', 'val')
    images_batch = np.random.choice(images, batch_size)
    labels_batch = np.random.choice(labels, batch_size)

    print('Validation images: {}, labels: {}'.format(len(images_batch), len(labels_batch)))
    images = []
    gt_images = []
    for image_file, gt_image_file in zip(images_batch, labels_batch):
        image = cv2.resize(cv2.imread(image_file), image_shape, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = (image.astype(np.float32)-mean)/std

        gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)
        gt_image = cv2.resize(gt_image[:, :, 0], image_shape, interpolation=cv2.INTER_NEAREST)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)


def get_cityscapes_f_paths(cityscapes_dir, phase='train'):
    phase = phase.lower()
    assert phase in ['train', 'val', 'test'], 'phase must in train val or test'
    # get all images and mask label file path
    labels_path = glob(os.path.join(cityscapes_dir, 'gtFine', phase, '*/*_gtFine_instanceIds.png'))
    images_path = glob(os.path.join(cityscapes_dir, 'leftImg8bit', phase, '*/*.png'))

    labels_path = sorted(labels_path)
    images_path = sorted(images_path)
    if len(labels_path) != len(images_path):
        print('images and labels are not equal there must be something wrong. {} vs {}'.format(
            len(images_path), len(labels_path)
        ))
        exit(0)
    else:
        return images_path, labels_path


def get_lane_f_paths(lane_list_f):
    # if using train list then read the text load all images and labels
    with open(lane_list_f, 'r') as f:
        image_files = []
        gt_files = []
        for l in f.readlines():
            l = l.strip().split(' ')
            image_files.append(l[0])
            gt_files.append(l[-2])
    log.info('Find {} images, {} gts.'.format(len(image_files), len(gt_files)))
    return image_files, gt_files


if __name__ == "__main__":
    pass
