import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(50, 8))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 4

ax1 = plt.subplot(151)
ax2 = plt.subplot(152)
ax3 = plt.subplot(153)
ax4 = plt.subplot(154)
ax5 = plt.subplot(155)

# ax6 = plt.subplot(256)
# ax7 = plt.subplot(257)
# ax8 = plt.subplot(258)
# ax9 = plt.subplot(259)
# ax10 = plt.subplot(2, 5, 10)

linewidth = 8
font = 48

markers = ['o', 's', '^']
colors = ['#edb03d', "#4dbeeb", "#77ac41", "#9b59b6"]
index_0 = [0, 1, 2, 3, 4, 5]
index = [0, 1, 2, 3, 4]
index3 = [0, 1, 2, 3, 4]
x_cpt_mnist = ["5", "10", "20", "50", "200", "300"]
x_1 = [0, 0.1, 1, 2, 5]
x_2 = [0, 0.1, 1, 5, 10]
x_3 = [0, 0.1, 1, 5, 10]
x_4 = [0, 0.1, 1, 2, 5]

ax1.set_xticks(index_0)
ax1.set_xticklabels(x_cpt_mnist)
ax2.set_xticks(index)
ax2.set_xticklabels(x_1)
ax3.set_xticks(index)
ax3.set_xticklabels(x_2)
ax4.set_xticks(index)
ax4.set_xticklabels(x_3)
ax5.set_xticks(index3)
ax5.set_xticklabels(x_4)
acc_1 = [0.114, 0.902, 0.962, 0.959, 0.970, 0.101]
inter_1 = [0.994, 0.912, 0.932, 0.887, 0.892]
exter_1 = [0.999, 0.902, 0.780, 0.913, 0.820]

acc_2 = [0.950, 0.962, 0.932, 0.886, 0.180]
inter_2 = [0.921, 0.932, 0.950, 0.938, 0.987]
exter_2 = [0.771, 0.780, 0.718, 0.736, 0.991]

acc_3 = [0.960, 0.956, 0.962, 0.955, 0.963]
inter_3 = [0.930, 0.926, 0.932, 0.945, 0.970]
exter_3 = [0.788, 0.790, 0.780, 0.755, 0.770]

acc_4 = [0.957, 0.967, 0.962, 0.948, 0.930]
inter_4 = [0.931, 0.935, 0.932, 0.938, 0.949]
exter_4 = [0.779, 0.793, 0.780, 0.766, 0.740]

acc_5 = [0.947, 0.959, 0.962, 0.938, 0.901]
inter_5 = [0.948, 0.935, 0.932, 0.938, 0.914]
exter_5 = [0.779, 0.773, 0.780, 0.871, 0.890]

acc_6 = [0.043, 0.328, 0.762, 0.798, 0.827, 0.015]
inter_6 = [0.954, 0.925, 0.910, 0.861, 0.635]
exter_6 = [0.192, 0.368, 0.375, 0.438, 0.871]

acc_7 = [0.710, 0.798, 0.778, 0.688, 0.328]
inter_7 = [0.892, 0.925, 0.905, 0.880, 0.622]
exter_7 = [0.280, 0.368, 0.388, 0.451, 0.753]

acc_8 = [0.732, 0.746, 0.798, 0.790, 0.782]
inter_8 = [0.303, 0.532, 0.925, 0.937, 0.961]
exter_8 = [0.456, 0.373, 0.368, 0.398, 0.482]

acc_9 = [0.704, 0.685, 0.798, 0.782, 0.775]
inter_9 = [0.905, 0.892, 0.925, 0.921, 0.934]
exter_9 = [0.851, 0.823, 0.368, 0.328, 0.179]

acc_10 = [0.052, 0.786, 0.798, 0.770, 0.782]
inter_10 = [0.993, 0.934, 0.925, 0.911, 0.802]
exter_10 = [0.500, 0.348, 0.368, 0.378, 0.361]

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

ax1.tick_params(labelsize=font+5)
ax2.tick_params(labelsize=font+5)
ax3.tick_params(labelsize=font+5)
ax4.tick_params(labelsize=font+5)
ax5.tick_params(labelsize=font+5)


size_1 = 19

ax1.set_xlabel("k", fontsize=font+size_1+5)
ax2.set_xlabel("$\lambda_{qua}$", fontsize=font+size_1)
ax3.set_xlabel("$\lambda_{con}$", fontsize=font+size_1)
ax4.set_xlabel("$\lambda_{dis}$", fontsize=font+size_1)
ax5.set_xlabel("$\lambda_R$", fontsize=font+size_1)
#
mk = 22
#
ax1.plot(index_0, acc_1, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="-", label="MNIST")
# ax1.plot(index, inter_1, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax1.plot(index, exter_1, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax2.plot(index, acc_2, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="-", label="MNIST")
# ax2.plot(index, inter_2, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax2.plot(index, exter_2, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax3.plot(index, acc_3, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="-", label="MNIST")
# ax3.plot(index, inter_3, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax3.plot(index, exter_3, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax4.plot(index, acc_4, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="-", label="MNIST")
# ax4.plot(index, inter_4, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax4.plot(index, exter_4, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax5.plot(index, acc_5, marker=markers[0], markevery=1, markersize=mk, color=colors[0], linewidth=linewidth, linestyle="-", label="MNIST")
# ax5.plot(index, inter_5, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax5.plot(index, exter_5, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")


ax1.plot(index_0, acc_6, marker=markers[1], markevery=1, markersize=mk, color=colors[3], linewidth=linewidth, linestyle="-", label="CUB200")
# ax6.plot(index, inter_6, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax6.plot(index, exter_6, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax2.plot(index, acc_7, marker=markers[1], markevery=1, markersize=mk, color=colors[3], linewidth=linewidth, linestyle="-", label="CUB200")
# ax7.plot(index, inter_7, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax7.plot(index, exter_7, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax3.plot(index, acc_8, marker=markers[1], markevery=1, markersize=mk, color=colors[3], linewidth=linewidth, linestyle="-", label="CUB200")
# ax8.plot(index, inter_8, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax8.plot(index, exter_8, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax4.plot(index, acc_9, marker=markers[1], markevery=1, markersize=mk, color=colors[3], linewidth=linewidth, linestyle="-", label="CUB200")
# ax9.plot(index, inter_9, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax9.plot(index, exter_9, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")
ax5.plot(index, acc_10, marker=markers[1], markevery=1, markersize=mk, color=colors[3], linewidth=linewidth, linestyle="-", label="CUB200")
# ax10.plot(index, inter_10, marker=markers[1], markevery=1, markersize=mk, color=colors[1], linewidth=linewidth, linestyle="-")
# ax10.plot(index, exter_10, marker=markers[2], markevery=1, markersize=mk, color=colors[2], linewidth=linewidth, linestyle="-")

pp = 45
ax1.legend(loc='lower center', fontsize=pp, ncol=1)
ax2.legend(loc='lower center', fontsize=pp, ncol=1)
ax3.legend(loc='lower center', fontsize=pp, ncol=1)
ax4.legend(loc='lower center', fontsize=pp, ncol=1)
ax5.legend(loc='lower center', fontsize=pp, ncol=1)
# ax1.grid(True)
# ax2.grid()
# ax3.grid()
# ax4.grid()
# ax5.grid()
# ax6.grid()
# ax7.grid()
# ax8.grid()
# ax9.grid()
# ax10.grid()

plt.tight_layout()
plt.savefig("ablation.pdf")
plt.show()