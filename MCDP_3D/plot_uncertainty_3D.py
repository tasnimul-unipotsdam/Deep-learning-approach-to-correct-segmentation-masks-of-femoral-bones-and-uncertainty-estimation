import numpy as np
import matplotlib.pyplot as plt

X_ts = np.load("D:/PROJECTS/internship/3D_data/test_input_image.npy")
Y_ts = np.load("D:/PROJECTS/internship/3D_data/test_label_image.npy")
Y_ts_hat = np.load("D:/PROJECTS/internship/MCDP_3D/Y_ts_hat_3D.npy")
U_ts = np.load("D:/PROJECTS/internship/MCDP_3D/U_ts_3D.npy")
dice = np.load("D:/PROJECTS/internship/MCDP_3D/dice_mcdp_3D.npy")

font_size = 30

threshold = 0.5

# image_index = [0, 2, 6, 9, 12,  15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
# 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 46, 47, 50]

image_index = [0, 36, 19, 27, 28]
X_ts = X_ts[image_index]
Y_ts = Y_ts[image_index]
Y_ts_hat = Y_ts_hat[image_index]
U_ts = U_ts[image_index]
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
    fig, axes = plt.subplots(Ntest, 5, figsize=(45, 25))
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

        axes[i, 3].imshow(U_ts[i, :, 95, :, 0])
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        axes[i, 3].set_title('Model Uncertainty', {'fontsize': font_size}, fontweight='bold')

        axes[i, 4].imshow(Y_ts_hat[i, :, 95, :, 0])
        axes[i, 4].imshow(U_ts[i, :, 95, :, 0], alpha=.5)
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])
        axes[i, 4].set_title('Prediction & Uncertainty Overlay', {'fontsize': font_size},
                             fontweight='bold')

    plt.savefig("D:/PROJECTS/internship/MCDP_3D/uncertainty_plot_3D.jpg", dpi=300)


if __name__ == '__main__':
    plot()

