import os
import random
import numpy as np
import tensorflow as tf
import pickle
from pipeline.tfrecord_reader import TFRecordReader
from model.unet import unet
from model.metrics_optimizer import *
from model.plot_metrics_loss import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

seeds = 1000
seed_everything(seeds)

train_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/records",
                               is_training=True).train_dataset()
validation_dataset = TFRecordReader("D:/PROJECTS/internship/2D_Data/records",
                                    is_training=False).validation_dataset()


def compile_model():
    model = unet()
    model.summary()
    model.compile(optimizer=adam,
                  loss=bce,
                  metrics=[dice_coefficient, robust_hausdorff], run_eagerly=True)
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

    model.save("D:/PROJECTS/internship/2D_Data/saved_model/unet_bce_model.h5")

    with open("D:/PROJECTS/internship/2D_data/saved_history/unet_bce_history.pickle",
              'wb') as hist_file:
        pickle.dump(history.history, hist_file)

    plot_loss_metrics(history, "D:/PROJECTS/internship/2D_Data/saved_plot/unet_bce_plot.jpg")


if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass
