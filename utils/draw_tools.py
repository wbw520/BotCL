import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 2


def draw_bar(data, name):
    plt.figure(figsize=(10, 6), dpi=80)
    font = 22
    x_bar = np.arange(0, len(data), 1)
    plt.bar(x_bar, data)
    plt.ylabel('Weight', fontsize=font)
    plt.xlabel('Concepts', fontsize=font)
    plt.xticks(list(x_bar))
    plt.tick_params(labelsize=22)
    plt.tight_layout()
    plt.savefig(name + "weight.pdf", bbox_inches="tight")
    plt.show()


def draw_plot(data, name):
    font = 22
    plt.figure(figsize=(10, 6), dpi=80)
    b, c = data.shape
    for i in range(c):
        plt.boxplot(data[:, i], positions=[i*10], widths=5, showmeans=True)

    plt.ylabel('Activation', fontsize=font)
    plt.xlabel('Concepts', fontsize=font)
    plt.xticks(np.arange(0, c*10, 10), list(np.arange(0, c, 1)))
    plt.tick_params(labelsize=22)
    plt.tight_layout()
    plt.savefig(name + "_heat.pdf", bbox_inches="tight")
    plt.show()

