import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['axes.linewidth'] = 2
x_label = []
for i in range(20):
    x_label.append("Cpt." + str(i + 1))

x_label2 = []
for i in range(10):
    x_label2.append("Cpt." + str(i + 1))

mnist_7 = [0.001, 0.001, 0.411, 0.001, 0.001, 0.001, 0.001, 0.62, 0.001, 0.001, 0.05, 0.16, 0.001, 0.001, 0.02, 0.011, 0.001, 0.001, 0.001, 0.001]
bird = [0.001, 0.601, 0.001, 0.001, 0.001, 0.101, 0.001, 0.29, 0.201, 0.421, 0.001, 0.001, 0.081, 0.001, 0.002, 0.501, 0.001, 0.001, 0.051, 0.301]
fish = [0.001, 0.501, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.421, 0.121]


def draw_hist():
    plt.figure(figsize=(8, 10), dpi=80)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.tick_params(labelsize=18)
    plt.rcParams['axes.linewidth'] = 5

    plt.barh(range(len(x_label)), mnist_7[::-1], height=0.6, color="orange")
    plt.yticks(range(len(x_label)), x_label[::-1], fontsize=22)
    plt.xticks(np.linspace(0, 1, 2, endpoint=True))
    plt.xlabel("Importance", fontsize=25)

    plt.tight_layout()
    plt.savefig("7.pdf")
    plt.show()


def draw_hist2():
    plt.figure(figsize=(25, 4), dpi=80)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.tick_params(labelsize=18)
    plt.rcParams['axes.linewidth'] = 5

    plt.bar(range(len(x_label)), bird, width=0.6, color="orange")
    plt.xticks(range(len(x_label)), x_label, fontsize=22)
    plt.yticks(np.linspace(0, 1, 2, endpoint=True))
    plt.ylabel("Importance", fontsize=30)
    plt.title("Importance of each concept for input image", fontsize=25)

    plt.tight_layout()
    plt.savefig("bird.png")
    plt.show()


def draw_hist3():
    plt.figure(figsize=(18, 4), dpi=80)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.tick_params(labelsize=18)
    plt.rcParams['axes.linewidth'] = 5

    plt.bar(range(len(x_label2)), fish, width=0.6, color="orange")
    plt.xticks(range(len(x_label2)), x_label2, fontsize=22)
    plt.yticks(np.linspace(0, 1, 2, endpoint=True))
    plt.ylabel("Importance", fontsize=30)
    plt.title("Importance of each concept for input image", fontsize=25)

    plt.tight_layout()
    plt.savefig("fish.png")
    plt.show()


draw_hist()