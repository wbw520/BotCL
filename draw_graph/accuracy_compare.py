import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(55, 10))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 4

ax1 = plt.subplot(141)
ax2 = plt.subplot(142)
ax3 = plt.subplot(143)
ax4 = plt.subplot(144)

linewidth = 9

font = 55
markers = ['o', 's', '']
colors = ['#edb03d', "#4dbeeb", "#77ac41"]
index = [0, 1, 2, 3, 4]

x_bird = ["20", "50", "100", "150", "200"]
x_imagenet = ["20", "50", "100", "150", "200"]

cub_base_18 = [0.914, 0.794, 0.772, 0.728, 0.685]
cub_ICLM_18 = [0.932, 0.805, 0.778, 0.735, 0.673]

cub_base_101 = [0.920, 0.868, 0.818, 0.795, 0.784]
cub_ICLM_101 = [0.924, 0.879, 0.829, 0.785, 0.771]

image_base_18 = [0.901, 0.825, 0.801, 0.786, 0.746]
image_ICLM_18 = [0.915, 0.823, 0.811, 0.790, 0.735]

image_base_101 = [0.932, 0.858, 0.814, 0.802, 0.794]
image_ICLM_101 = [0.940, 0.867, 0.820, 0.815, 0.785]


ax1.set_xticks(index)
ax1.set_xticklabels(x_bird)
ax2.set_xticks(index)
ax2.set_xticklabels(x_bird)
ax3.set_xticks(index)
ax3.set_xticklabels(x_imagenet)
ax4.set_xticks(index)
ax4.set_xticklabels(x_imagenet)

ax1.axis(ymin=0.6, ymax=1.0)
ax1.set_yticks(np.linspace(0.6, 1.0, 5, endpoint=True))
ax2.axis(ymin=0.6, ymax=1.0)
ax2.set_yticks([])

ax3.axis(ymin=0.6, ymax=1.0)
ax3.set_yticks([])
ax4.axis(ymin=0.6, ymax=1.0)
ax4.set_yticks([])

ax1.tick_params(labelsize=font+5)
ax2.tick_params(labelsize=font+5)
ax3.tick_params(labelsize=font+5)
ax4.tick_params(labelsize=font+5)

ax1.set_title("CUB ResNet-18", fontsize=font+8)
ax2.set_title("CUB ResNet-101", fontsize=font+8)
ax3.set_title("ImageNet ResNet-18", fontsize=font+8)
ax4.set_title("ImageNet ResNet-101", fontsize=font+8)

ax1.plot(index, cub_base_18, marker=markers[0], markevery=1, markersize=30, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax1.plot(index, cub_ICLM_18, marker=markers[1], markevery=1, markersize=30, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax1.legend(loc='upper right', fontsize=font, ncol=1)

ax2.plot(index, cub_base_101, marker=markers[0], markevery=1, markersize=30, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax2.plot(index, cub_ICLM_101, marker=markers[1], markevery=1, markersize=30, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax2.legend(loc='upper right', fontsize=font, ncol=1)

ax3.plot(index, image_base_18, marker=markers[0], markevery=1, markersize=30, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax3.plot(index, image_ICLM_18, marker=markers[1], markevery=1, markersize=30, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax3.legend(loc='upper right', fontsize=font, ncol=1)

ax4.plot(index, image_base_101, marker=markers[0], markevery=1, markersize=30, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax4.plot(index, image_ICLM_101, marker=markers[1], markevery=1, markersize=30, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax4.legend(loc='upper right', fontsize=font, ncol=1)

plt.tight_layout()
plt.savefig("acc_compare.pdf")
plt.show()