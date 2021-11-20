import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(36, 10), dpi=80)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 3

ax1 = plt.subplot(251)
ax2 = plt.subplot(252)
ax3 = plt.subplot(253)
ax4 = plt.subplot(254)
ax5 = plt.subplot(255)

ax6 = plt.subplot(256)
ax7 = plt.subplot(257)
ax8 = plt.subplot(258)
ax9 = plt.subplot(259)
ax10 = plt.subplot(2, 5, 10)

linewidth = 5
font = 30

markers = ['o', 's', '^']
colors = ['#edb03d', "#4dbeeb", "#77ac41"]
index = [0, 1, 2, 3, 4]
index3 = [0, 1, 2, 3, 4]
x_cpt_mnist = ["5", "10", "20", "50", "100"]
x_cpt_bird = ["20", "50", "100", "200", "300"]
x_1 = [0, 0.1, 1, 2, 5]
x_2 = [0, 0.1, 1, 5, 10]
x_3 = [0, 0.1, 1, 5, 10]
x_4 = [0, 0.1, 1, 2, 5]

ax1.set_xticks(index)
ax1.set_xticklabels(x_cpt_mnist)
ax2.set_xticks(index)
ax2.set_xticklabels(x_1)
ax3.set_xticks(index)
ax3.set_xticklabels(x_2)
ax4.set_xticks(index)
ax4.set_xticklabels(x_3)
ax5.set_xticks(index3)
ax5.set_xticklabels(x_4)
acc_1 = [0.857, 0.922, 0.962, 0.995, 0.992]
inter_1 = [0.732, 0.791, 0.930, 0.965, 0.988]
exter_1 = [0.531, 0.513, 0.618, 0.906, 0.960]

acc_2 = [0.940, 0.962, 0.922, 0.886, 0.780]
inter_2 = [0.921, 0.930, 0.930, 0.938, 0.889]
exter_2 = [0.731, 0.723, 0.718, 0.736, 0.849]

acc_3 = [0.960, 0.956, 0.962, 0.955, 0.963]
inter_3 = [0.930, 0.926, 0.930, 0.945, 0.970]
exter_3 = [0.711, 0.710, 0.718, 0.715, 0.789]

acc_4 = [0.957, 0.967, 0.962, 0.948, 0.900]
inter_4 = [0.931, 0.935, 0.930, 0.938, 0.949]
exter_4 = [0.719, 0.713, 0.718, 0.716, 0.700]

acc_5 = [0.927, 0.949, 0.962, 0.948, 0.952]
inter_5 = [0.981, 0.935, 0.930, 0.948, 0.949]
exter_5 = [0.979, 0.923, 0.718, 0.746, 0.750]

ax6.set_xticks(index)
ax6.set_xticklabels(x_cpt_bird)
ax7.set_xticks(index)
ax7.set_xticklabels(x_1)
ax8.set_xticks(index)
ax8.set_xticklabels(x_2)
ax9.set_xticks(index)
ax9.set_xticklabels(x_3)
ax10.set_xticks(index3)
ax10.set_xticklabels(x_4)

acc_6 = [0.566, 0.668, 0.675, 0.680, 0.676]
inter_6 = [0.702, 0.850, 0.930, 0.961, 0.965]
exter_6 = [0.631, 0.353, 0.245, 0.248, 0.361]

acc_7 = [0.650, 0.668, 0.642, 0.588, 0.128]
inter_7 = [0.752, 0.850, 0.880, 0.810, 0.402]
exter_7 = [0.601, 0.353, 0.332, 0.371, 0.853]

acc_8 = [0.704, 0.685, 0.668, 0.662, 0.634]
inter_8 = [0.652, 0.812, 0.850, 0.921, 0.960]
exter_8 = [0.381, 0.373, 0.345, 0.398, 0.401]

acc_9 = [0.620, 0.656, 0.668, 0.660, 0.652]
inter_9 = [0.812, 0.823, 0.850, 0.881, 0.880]
exter_9 = [0.671, 0.583, 0.345, 0.298, 0.291]

acc_10 = [0.670, 0.656, 0.668, 0.660, 0.652]
inter_10 = [0.782, 0.799, 0.890, 0.891, 0.900]
exter_10 = [0.361, 0.343, 0.265, 0.268, 0.261]

ax1.axis(ymin=0, ymax=1)
ax2.axis(ymin=0, ymax=1)
ax3.axis(ymin=0, ymax=1)
ax4.axis(ymin=0, ymax=1)
ax5.axis(ymin=0, ymax=1)
ax1.set_yticks(np.linspace(0, 1, 2, endpoint=True))
ax2.set_yticks([])
ax3.set_yticks([])
ax4.set_yticks([])
ax5.set_yticks([])

ax6.axis(ymin=0, ymax=1)
ax7.axis(ymin=0, ymax=1)
ax8.axis(ymin=0, ymax=1)
ax9.axis(ymin=0, ymax=1)
ax10.axis(ymin=0, ymax=1)
ax6.set_yticks(np.linspace(0, 1, 2, endpoint=True))
ax7.set_yticks([])
ax8.set_yticks([])
ax9.set_yticks([])
ax10.set_yticks([])

ax1.tick_params(labelsize=font+5)
ax2.tick_params(labelsize=font+5)
ax3.tick_params(labelsize=font+5)
ax4.tick_params(labelsize=font+5)
ax5.tick_params(labelsize=font+5)
ax6.tick_params(labelsize=font+5)
ax7.tick_params(labelsize=font+5)
ax8.tick_params(labelsize=font+5)
ax9.tick_params(labelsize=font+5)
ax10.tick_params(labelsize=font+5)


size_1 = 10

ax1.set_ylabel("MNIST", fontsize=font+size_1)
ax6.set_ylabel("CUB200", fontsize=font+size_1)
ax6.set_xlabel("k", fontsize=font+size_1+5)
ax7.set_xlabel("$\lambda_{qua}$", fontsize=font+size_1)
ax8.set_xlabel("$\lambda_{con}$", fontsize=font+size_1)
ax9.set_xlabel("$\lambda_{dis}$", fontsize=font+size_1)
ax10.set_xlabel("$\lambda_R$", fontsize=font+size_1)

ax1.plot(index, acc_1, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax1.plot(index, inter_1, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax1.plot(index, exter_1, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax2.plot(index, acc_2, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax2.plot(index, inter_2, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax2.plot(index, exter_2, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax3.plot(index, acc_3, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax3.plot(index, inter_3, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax3.plot(index, exter_3, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax4.plot(index, acc_4, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax4.plot(index, inter_4, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax4.plot(index, exter_4, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax5.plot(index, acc_5, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax5.plot(index, inter_5, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax5.plot(index, exter_5, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")


ax6.plot(index, acc_6, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax6.plot(index, inter_6, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax6.plot(index, exter_6, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax7.plot(index, acc_7, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax7.plot(index, inter_7, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax7.plot(index, exter_7, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax8.plot(index, acc_8, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax8.plot(index, inter_8, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax8.plot(index, exter_8, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax9.plot(index, acc_9, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax9.plot(index, inter_9, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax9.plot(index, exter_9, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")
ax10.plot(index, acc_10, marker=markers[0], markevery=1, markersize=15, color=colors[0], linewidth=linewidth, linestyle="-")
ax10.plot(index, inter_10, marker=markers[1], markevery=1, markersize=15, color=colors[1], linewidth=linewidth, linestyle="-")
ax10.plot(index, exter_10, marker=markers[2], markevery=1, markersize=15, color=colors[2], linewidth=linewidth, linestyle="-")


plt.tight_layout()
plt.savefig("ablation.pdf")
plt.show()