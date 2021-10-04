import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from distance import compute_surface_distances, compute_average_surface_distance, \
    compute_robust_hausdorff

smooth = 1


def robust_hausdorff(mask_gt, mask_pred):
    mask_gt_cp = mask_gt.numpy().copy()
    mask_pred_cp = mask_pred.numpy().copy()

    haus_dorff_dist = []
    for i in range(mask_gt.shape[0]):
        m_gt = mask_gt_cp[i]
        p_gt = mask_pred_cp[i]

        m_gt = np.asarray(m_gt, np.float32)
        m_gt = m_gt >= 0.5
        m_gt = np.resize(m_gt, (mask_gt.shape[1], mask_gt.shape[2]))
        # print(m_gt.shape)

        p_gt = np.asarray(p_gt, np.float32)
        p_gt = p_gt >= 0.5
        p_gt = np.resize(p_gt, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(m_gt, p_gt, spacing_mm=(1, 1))
        haus_dorff_dist.append(compute_robust_hausdorff(surface_dict, 100))

    avg = sum(haus_dorff_dist) / 4
    return avg


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def surface_distance(mask_gt, mask_pred):
    mask_gt_cp = mask_gt.numpy().copy()
    mask_pred_cp = mask_pred.numpy().copy()

    sf_list = []
    for i in range(mask_gt.shape[0]):
        m_gt = mask_gt_cp[i]
        p_gt = mask_pred_cp[i]

        m_gt = np.asarray(m_gt, np.float32)
        m_gt = m_gt >= 0.5
        m_gt = np.resize(m_gt, (mask_gt.shape[1], mask_gt.shape[2]))

        p_gt = np.asarray(p_gt, np.float32)
        p_gt = p_gt >= 0.5
        p_gt = np.resize(p_gt, (mask_pred.shape[1], mask_pred.shape[2]))

        surface_dict = compute_surface_distances(m_gt, p_gt, spacing_mm=(1, 1))
        sf = compute_average_surface_distance(surface_dict)
        sf_list.append([sf[0], sf[1]])

    sum_arr = np.array(sf_list).sum(axis=0)
    avg = sum_arr / 4
    return tuple(avg)
