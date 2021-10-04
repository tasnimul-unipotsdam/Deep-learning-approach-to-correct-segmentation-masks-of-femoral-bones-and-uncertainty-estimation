import numpy as np
import matplotlib.pyplot as plt

Y_ts = np.load("D:/PROJECTS/internship/MCDP/Y_ts.npy")
X_ts = np.load("D:/PROJECTS/internship/MCDP/X_ts.npy")
Y_ts_hat = np.load("D:/PROJECTS/internship/MCDP/Y_ts_hat.npy")
U_ts = np.load("D:/PROJECTS/internship/MCDP/U_ts.npy")
dice = np.load("D:/PROJECTS/internship/MCDP/dice.npy")

image_index = [65, 421, 550, 869, 1100]
X_ts = X_ts[image_index]
Y_ts = Y_ts[image_index]
Y_ts_hat = Y_ts_hat[image_index]
U_ts = U_ts[image_index]
dice = dice[image_index]

Ntest = len(X_ts)
print(Ntest)


font_size = 30
threshold = 0.5

Y_ts_hat[Y_ts_hat < threshold] = 0
Y_ts_hat[Y_ts_hat >= threshold] = 1


def plot():
    fig, axes = plt.subplots(Ntest, 5, figsize=(45, 25))
    for i in range(Ntest):
        axes[i, 0].imshow(X_ts[i, :, :,  0])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('input image', {'fontsize': font_size}, fontweight='bold')

        axes[i, 1].imshow(Y_ts[i, :, :,  0])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title('True mask', {'fontsize': font_size}, fontweight='bold')

        axes[i, 2].imshow(Y_ts_hat[i, :, :,  0])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title('Pred mask, dice=' + str(np.round(dice[i], 2)),
                             {'fontsize': font_size}, fontweight='bold')

        axes[i, 3].imshow(U_ts[i, :, :,  0])
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        axes[i, 3].set_title('Model Uncertainty', {'fontsize': font_size}, fontweight='bold')

        axes[i, 4].imshow(Y_ts_hat[i, :, :,  0])
        axes[i, 4].imshow(U_ts[i, :, :,  0], alpha=.5)
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])
        axes[i, 4].set_title('Prediction & Uncertainty Overlay', {'fontsize': font_size},
                             fontweight='bold')

    plt.savefig("D:/PROJECTS/internship/MCDP/uncertainty_plot_2D.jpg", dpi=300)


if __name__ == '__main__':
    plot()
    pass

