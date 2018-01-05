import os
import numpy as np
import matplotlib.pyplot as plt


image_path = os.path.join(os.path.dirname(__file__), '../images/plot_data.png')


def plot_data(data):
    plotting = data.reshape((data.shape[0], 28, 28))
    fig = plt.figure(figsize=(8, 3))
    for i in range(10):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        img = plotting[i]
        plt.imshow(img, cmap="gray")
    plt.savefig(image_path)
