import matplotlib.pyplot as plt


def plot_crossentropy_loss(history, name):
    fig = plt.figure()

    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('crossentropy_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_dice_loss(history, name):
    fig = plt.figure()

    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('dice loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_dice_coefficient(history, name):
    fig = plt.figure()

    plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_hausdorff(history, name):
    fig = plt.figure()

    plt.plot(history.history['robust_hausdorff'], color='b', label="train_robust_hausdorff")
    plt.plot(history.history['val_robust_hausdorff'], color='r',
             label="validation_robust_hausdorff")
    plt.title('hausdorff_distance')
    plt.ylabel('hausdorff_distance')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


# def plot_metrics(history, name):
#     fig = plt.figure(num=None, figsize=(20, 6), dpi=150, facecolor='w', edgecolor='k',
#                      tight_layout=True)
#
#     plt.subplot(1, 3, 1)
#     plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
#     plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
#     plt.title('dice coefficient')
#     plt.ylabel('coefficient')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'])
#
#     plt.subplot(1, 3, 2)
#     plt.plot(history.history['surface_distance'], color='b', label="train_surface_distance")
#     plt.plot(history.history['val_surface_distance'], color='r',
#              label="validation_surface_distance")
#     plt.title('surface distance')
#     plt.ylabel('surface_distance')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'])
#
#     plt.subplot(1, 3, 3)
#     plt.plot(history.history['robust_hausdorff'], color='b', label="train_robust_hausdorff")
#     plt.plot(history.history['val_robust_hausdorff'], color='r',
#              label="validation_robust_hausdorff")
#     plt.title('hausdorff distance')
#     plt.ylabel('hausdorff_distance')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'])
#
#     plt.show()
#     fig.savefig(name)


def plot_dice_loss_dice_coefficient_3D(history, name):
    fig = plt.figure(num=None, figsize=(20, 6), dpi=150, facecolor='w', edgecolor='k',
                     tight_layout=True)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('dice loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_crossentropy_loss_dice_coefficient_3D(history, name):
    fig = plt.figure(num=None, figsize=(20, 6), dpi=150, facecolor='w', edgecolor='k',
                     tight_layout=True)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('crossentropy_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)


def plot_loss_metrics(history, name):
    fig = plt.figure(num=None, figsize=(20, 6), dpi=150, facecolor='w', edgecolor='k',
                     tight_layout=True)

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.title('dice loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 3, 2)
    plt.plot(history.history['dice_coefficient'], color='b', label="train_coefficient")
    plt.plot(history.history['val_dice_coefficient'], color='r', label="validation_coefficient")
    plt.title('dice coefficient')
    plt.ylabel('coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 3, 3)
    plt.plot(history.history['robust_hausdorff'], color='b', label="train_robust_hausdorff")
    plt.plot(history.history['val_robust_hausdorff'], color='r', label="validation_robust_hausdorff")
    plt.title('hausdorff distance')
    plt.ylabel('hausdorff_distance')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.show()
    fig.savefig(name)
