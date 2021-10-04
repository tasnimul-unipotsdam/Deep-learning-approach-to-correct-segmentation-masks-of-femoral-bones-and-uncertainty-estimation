import numpy as np
import matplotlib.pyplot as plt

X_ts = np.load("D:/PROJECTS/internship/3D_data/test_input_image.npy")
Y_ts = np.load("D:/PROJECTS/internship/3D_data/test_label_image.npy")

Y_ts_hat = np.load("D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_pred_mask.npy")
dice = np.load("D:/PROJECTS/internship/3D_data/unet_3D_02/unet_3D_02_dice_score.npy")

font_size = 30
threshold = 0.5
image_index = [34, 36, 19, 27, 28]

X_ts = X_ts[image_index]
Y_ts = Y_ts[image_index]
Y_ts_hat = Y_ts_hat[image_index]
dice = dice[image_index]

Ntest = len(X_ts)
print(Ntest)

X_ts[X_ts < threshold] = 0
X_ts[X_ts >= threshold] = 1
Y_ts[Y_ts < threshold] = 0
Y_ts[Y_ts >= threshold] = 1
Y_ts_hat[Y_ts_hat < threshold] = 0
Y_ts_hat[Y_ts_hat >= threshold] = 1


def plot():
    fig, axes = plt.subplots(Ntest, 3, figsize=(15, 15))
    for i in range(Ntest):
        axes[i, 0].imshow(X_ts[i, :, 95, :, 0])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('input image', {'fontsize': font_size}, fontweight='bold')

        axes[i, 1].imshow(Y_ts[i, :, 95, :, 0])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title('True mask', {'fontsize': font_size}, fontweight='bold')

        axes[i, 2].imshow(Y_ts_hat[i, :, 95, :, 0])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title('Pred mask, dice=' + str(np.round(dice[i], 2)),
                             {'fontsize': font_size}, fontweight='bold')

    plt.savefig("D:/PROJECTS/internship/pred_plot_unet3d02.jpg")


if __name__ == '__main__':
    plot()
