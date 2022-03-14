import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(15, 10))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 4

ax1 = plt.subplot(111)

linewidth = 7

font = 35
mk = 16
pp = 35
markers = ['o', 's', '^']
colors = ['#edb03d', "#4dbeeb", "#77ac41", "#9b59b6", "#4848fd", "#4ff519"]
index1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
index2 = [0, 1, 2, 3, 4, 5, 6, 7]

x_cpt = ["5", "10", "20", "50", "100", "150", "200", "250", "300"]
x_imagenet = ["20", "50", "100", "150", "200", "300", "400", "500"]

base_18 = [0.901, 0.865, 0.821, 0.802, 0.772, 0.760, 0.746, 0.728]
image_ICLM_18 = [0.915, 0.863, 0.831, 0.799, 0.766, 0.750, 0.728, 0.702]
base_101 = [0.932, 0.888, 0.874, 0.862, 0.830, 0.816, 0.808, 0.802]
image_ICLM_101 = [0.940, 0.893, 0.875, 0.860, 0.828, 0.804, 0.795, 0.785]

acc_MNIST = [0.114, 0.902, 0.962, 0.959, 0.970, 0.968, 0.972, 0.133, 0.101]
acc_BIRD = [0.043, 0.328, 0.762, 0.798, 0.805, 0.829, 0.827, 0.326, 0.015]
acc_Image = [0.063, 0.528, 0.832, 0.864, 0.875, 0.872, 0.865, 0.526, 0.025]

ax1.set_xticks(index2)
ax1.set_xticklabels(x_imagenet)
ax1.axis(ymin=0.6, ymax=1)
ax1.set_yticks(np.linspace(0.6, 1, 2, endpoint=True))
ax1.tick_params(labelsize=font+5)
# ax1.set_xlabel("Class number n", fontsize=font+5)

ax1.plot(index2, base_18, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="-", label="Res-18")
ax1.plot(index2, image_ICLM_18, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="dashed", label="BotCL-18")
ax1.plot(index2, base_101, marker=markers[0], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-", label="Res-101")
ax1.plot(index2, image_ICLM_101, marker=markers[0], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="dashed", label="BotCL-101")
ax1.legend(loc='lower left', fontsize=pp, ncol=1)

# ax2 = plt.subplot(111)
# ax2.set_xticks(index1)
# ax2.set_xticklabels(x_cpt)
# ax2.axis(ymin=0, ymax=1)
# ax2.set_yticks(np.linspace(0, 1, 2, endpoint=True))
# ax2.tick_params(labelsize=font+5)
# # ax2.set_xlabel("Concept number k", fontsize=font+5)
#
# ax1.plot(index1, acc_MNIST, marker=markers[0], markevery=1, markersize=mk, color=colors[3], linewidth=linewidth, linestyle="-", label="MNIST")
# ax1.plot(index1, acc_BIRD, marker=markers[1], markevery=1, markersize=mk, color=colors[4], linewidth=linewidth, linestyle="-", label="CUB200")
# ax1.plot(index1, acc_Image, marker=markers[2], markevery=1, markersize=mk, color=colors[5], linewidth=linewidth, linestyle="-", label="ImageNet")
# ax2.legend(loc='lower center', fontsize=pp, ncol=1)


plt.tight_layout()
plt.savefig("spp_acc_class.pdf")
plt.show()