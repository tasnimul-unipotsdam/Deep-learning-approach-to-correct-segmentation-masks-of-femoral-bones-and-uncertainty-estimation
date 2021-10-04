import os
import random

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from distance import compute_surface_distances, compute_robust_hausdorff
from test_tfrecord_reader import TFRecordReader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seeds = 1000
seed_everything(seeds)


def test_data():
    test_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/test_records", shuffle=False,
                                  batch_size=4).test_dataset()
    test_image = np.concatenate([x for x, y in test_dataset], axis=0)
    test_label = np.concatenate([y for x, y in test_dataset], axis=0)
    np.save("D:/PROJECTS/internship/2D_Data/test_numpy/test_image.npy", test_image)
    np.save("D:/PROJECTS/internship/2D_Data/test_numpy/test_label.npy", test_label)
    return test_image, test_label


def prediction():
    model = tf.keras.models.load_model(
        "D:/PROJECTS/internship/2D_Data/saved_model/unet_dice_model.h5",
        compile=False)
    test_image = np.load("D:/PROJECTS/internship/2D_Data/test_numpy/test_image.npy")

    mask_prediction = model.predict(test_image, batch_size=1)
    pred_mask = np.save("D:/PROJECTS/internship/2D_data/test_results/pred_model_unet_dice.npy",
                        mask_prediction)
    return pred_mask


def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def compute_dice():
    test_label = np.load("D:/PROJECTS/internship/2D_Data/test_numpy/test_label.npy")
    Ntest = len(test_label)
    pred_mask = np.load("D:/PROJECTS/internship/2D_data/test_results/pred_model_unet_dice.npy")
    dice = []
    for i in range(Ntest):
        dice.append(dice_coefficient(test_label[i, :, :, 0], pred_mask[i, :, :, 0]))
    dice = np.array(dice)
    np.save('D:/PROJECTS/internship/2D_data/test_results/dice_score_unet_dice.npy', dice)


def robust_hausdorff(m_gt, m_pr):
    m_gt = m_gt > 0.5
    m_gt = np.resize(m_gt, (m_gt.shape[0], m_gt.shape[1]))

    m_pr = m_pr > 0.5
    m_pr = np.resize(m_pr, (m_pr.shape[0], m_pr.shape[1]))

    surface_distance = compute_surface_distances(m_gt, m_pr, spacing_mm=(1, 1))
    hausdorff_distance = compute_robust_hausdorff(surface_distance, 100)

    return hausdorff_distance


def compute_hausdorff():
    test_label = np.load("D:/PROJECTS/internship/2D_Data/test_numpy/test_label.npy")
    Ntest = len(test_label)
    pred_mask = np.load("D:/PROJECTS/internship/2D_data/test_results/pred_model_unet_dice.npy")
    hausdorff = []
    for i in range(Ntest):
        hausdorff.append(robust_hausdorff(test_label[i, :, :, 0], pred_mask[i, :, :, 0]))
    hausdorff = np.array(hausdorff)
    np.save('D:/PROJECTS/internship/2D_data/test_results/hd_unet_dice.npy', hausdorff)


def confusion_matrix():
    smooth = 1
    pred_mask = np.load("D:/PROJECTS/internship/2D_data/test_results/pred_model_unet_dice.npy")
    test_label = np.load("D:/PROJECTS/internship/2D_Data/test_numpy/test_label.npy")

    y_pred_pos = K.round(K.clip(pred_mask, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(test_label, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    tn = K.sum(y_neg * y_pred_neg)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    cm_dict = {"precision": precision.numpy(),
               "recall": recall.numpy()}

    print(cm_dict)

    with open("D:/PROJECTS/internship/2D_Data/test_results/cm_dice.txt", 'w') as data:
        data.write(str(cm_dict))

    print(
        'True Positive: {}, False Positive : {}, False Negative: {}, True Negative: {}, '
        'Precision: {}, Recall: {} '.format(
            tp, fp, fn, tn, precision, recall))

    return precision, recall


if __name__ == '__main__':
    # test_data()
    # prediction()
    # compute_dice()
    # compute_hausdorff()
    confusion_matrix()
