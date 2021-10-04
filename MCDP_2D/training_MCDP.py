import os
import random
import numpy as np
import tensorflow as tf

from tfrecord_reader import TFRecordReader
from models import *
from metrics_optimizer import *
from plot_metrics_loss import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

seeds = 1000


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(seeds)

adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

train_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/records",
                               is_training=True).train_dataset()
validation_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/records",
                                    is_training=False).validation_dataset()


def compile_model():
    model = unet_mcdp_1()
    model.summary()
    model.compile(optimizer=adam,
                  loss=bce,
                  metrics=[dice_coefficient, robust_hausdorff],
                  run_eagerly=True)
    return model


def train_model(model):
    BATCH_SIZE = 4
    STEPS_PER_EPOCH = 1016 // BATCH_SIZE
    VALIDATION_STEPS = 254 // BATCH_SIZE
    history = model.fit(train_dataset,
                        verbose=1,
                        epochs=100,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=validation_dataset,
                        workers=12)

    model.save("D:/PROJECTS/internship/MCDP/saved_model/unet_bce_mcdp_model.h5")

    plot_loss_metrics(history, "D:/PROJECTS/internship/MCDP/saved_model/unet_bce_mcdp_plot.jpg")


if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass
