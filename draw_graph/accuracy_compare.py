import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(15, 10), dpi=80)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 3

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

linewidth = 5

font = 28
markers = ['o', 's', '']
colors = ['#edb03d', "#4dbeeb", "#77ac41"]
index = [0, 1, 2, 3, 4]

x_bird = ["20", "50", "100", "150", "200"]
x_imagenet = ["50", "100", "200", "250", "300"]

cub_base_18 = [0.862, 0.784, 0.752, 0.738, 0.695]
cub_ICLM_18 = [0.858, 0.780, 0.740, 0.719, 0.665]

cub_base_101 = [0.925, 0.858, 0.818, 0.805, 0.794]
cub_ICLM_101 = [0.910, 0.856, 0.809, 0.789, 0.762]

image_base_18 = [0.831, 0.792, 0.781, 0.762, 0.726]
image_ICLM_18 = [0.835, 0.801, 0.783, 0.751, 0.701]

image_base_101 = [0.858, 0.814, 0.804, 0.789, 0.761]
image_ICLM_101 = [0.864, 0.820, 0.798, 0.771, 0.720]


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

ax3.axis(ymin=0.65, ymax=0.9)
ax3.set_yticks(np.linspace(0.6, 0.9, 4, endpoint=True))
ax4.axis(ymin=0.65, ymax=0.9)
ax4.set_yticks([])

ax1.tick_params(labelsize=font+5)
ax2.tick_params(labelsize=font+5)
ax3.tick_params(labelsize=font+5)
ax4.tick_params(labelsize=font+5)

ax1.set_title("CUB ResNet-18", fontsize=font+8)
ax2.set_title("CUB ResNet-101", fontsize=font+8)
ax3.set_title("ImageNet ResNet-18", fontsize=font+8)
ax4.set_title("ImageNet ResNet-101", fontsize=font+8)

ax1.plot(index, cub_base_18, marker=markers[0], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax1.plot(index, cub_ICLM_18, marker=markers[1], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax1.legend(loc='upper right', fontsize=font-2, ncol=1)

ax2.plot(index, cub_base_101, marker=markers[0], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax2.plot(index, cub_ICLM_101, marker=markers[1], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax2.legend(loc='upper right', fontsize=font-2, ncol=1)

ax3.plot(index, image_base_18, marker=markers[0], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax3.plot(index, image_ICLM_18, marker=markers[1], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax3.legend(loc='upper right', fontsize=font-2, ncol=1)

ax4.plot(index, image_base_101, marker=markers[0], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-", label="Base")
ax4.plot(index, image_ICLM_101, marker=markers[1], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-", label="BotCL")
ax4.legend(loc='upper right', fontsize=font-2, ncol=1)

plt.tight_layout()
plt.savefig("acc_compare.pdf")
plt.show()