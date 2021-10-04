import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from random import randint

AUTOTUNE = tf.data.experimental.AUTOTUNE


def normalize(corrupt_image, original_image):
    corrupt_image = corrupt_image / 255
    original_image = original_image / 255
    return corrupt_image, original_image


def rotate(corrupt_image, original_image):
    angle = randint(-5, 5)
    radians = (angle / 180) * 3.14

    concat = tf.concat([corrupt_image, original_image], axis=2)
    concat = tfa.image.rotate(concat, radians)
    corrupt_image, original_image = tf.split(concat, num_or_size_splits=2, axis=2)
    return corrupt_image, original_image


class TFRecordReader(object):

    def __init__(self, record_path, is_training=True):
        self.record_path = record_path
        self.seed = 10
        self.batch_size = 4
        self.is_shuffle = True
        self.is_training = 'train' if is_training else 'validation'

        if self.is_training:
            self.data_type = 'train'
            self.buffer = 1000
        else:
            self.data_type = 'validation'
            self.buffer = 1000

    @classmethod
    def parse_record(cls, record):
        features = {
            'crr_image': tf.io.FixedLenFeature([], dtype=tf.string),
            'crr_img_height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'crr_img_width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'crr_img_depth': tf.io.FixedLenFeature([], dtype=tf.int64),

            'org_image': tf.io.FixedLenFeature([], dtype=tf.string),
            'org_img_height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'org_img_width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'org_img_depth': tf.io.FixedLenFeature([], dtype=tf.int64),
        }

        record = tf.io.parse_single_example(record, features)

        crr_image = tf.io.decode_raw(record['crr_image'], tf.float32)
        crr_image = tf.reshape(crr_image, [record['crr_img_height'], record['crr_img_width'],
                                           record['crr_img_depth']])

        org_image = tf.io.decode_raw(record['org_image'], tf.float32)
        org_image = tf.reshape(org_image, [record['org_img_height'], record['org_img_width'],
                                           record['org_img_depth']])

        return crr_image, org_image

    def train_dataset(self):
        files = os.path.join(self.record_path, f'{self.data_type}_femur.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=self.is_shuffle, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                                     cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(rotate, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def validation_dataset(self):
        files = os.path.join(self.record_path, f'{self.data_type}_femur.tfrecord')
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=self.is_shuffle, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn),
                                     cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
        # dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    train_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/records",
                                   is_training=True).train_dataset()
    validation_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/records",
                                        is_training=False).validation_dataset()
    print(train_dataset)

    for i, batch in enumerate(train_dataset):
        images, labels = batch
        print('images size: {}, labels size: {}'.format(images.shape, labels.shape))
        break

    image, label = next(iter(train_dataset))

    for data in train_dataset.take(1):
        img = np.resize(data[0].numpy()[2], (384, 160))
        lab = np.resize(data[1].numpy()[2], (384, 160))
        plt.imshow(img)
        plt.imshow(lab)
        plt.imsave('img.png', img)
        plt.imsave("lab.png", lab)
