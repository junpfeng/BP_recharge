# from pylab import *
# import matplotlib.pyplot as pyplot
# a = [ pow(10,i) for i in range(10) ]
# fig = pyplot.figure()
# ax = fig.add_subplot(2,1,1)
# line, = ax.plot(a, color='blue', lw=2)
# show()

import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

N = 576
K = 432
# N  = 16
# K = 8
# N = 128
# K = 64
# N = 32
# K = 16
# N = 96
# K = 48

BP_iter_num = 30

# list_BP_iter_num = [10, 20, 40, 50, 60] #
# list_BP_iter_num = [5, 10, 20, 40]
# list_BP_iter_num = [40, 50, 51]
# list_BP_iter_num = [20, 50, 21]  # BPDNN20, LLRBP50, LLRBP20
# list_BP_iter_num = [20, 50, 21]  # BPDNN20, LLRBP50, advance_BPDNN
list_BP_iter_num = [26, 50]
# list_BP_iter_num = [20, 50, 22, 51]  # BPDNN(10)-CNN-BPDNN(10)，BP(25)-CNN-BP(25), LLRBP50
# 20: BPDNN20
# 21: LLRBP20
# 22：BPDNN10-CNN-BPDNN10/advance_BPDNN
# 50: LLRBP50
# 51：BP(25)-CNN-BP(25)
# 线条配合形状，构建6种不同图例
list_marker = ["*", "", "", "", "o", "."]
list_line_style = ['-', '--', '-.', ':', '-', '-']

for i in range(len(list_BP_iter_num)):
    BP_iter_num = list_BP_iter_num[i]
    marker = list_marker[i]
    linestyle = list_line_style[i]
    # plot_file = format("model/data_back_up/BER(%s_%s)_BP(%s).txt" % (N, K, BP_iter_num))
    # plot_file = format("model/bp_model/plot_BPDNN/BER(%s_%s)_BP(%s)_train500.txt" % (N, K, BP_iter_num))
    # plot_file = format("model/bp_model/32_16/BP20/BER(%s_%s)_BP(%s)_train500.txt" % (N, K, BP_iter_num))
    plot_file = format("model/bp_model/plot_LLRBP/BER(%s_%s)_BP(%s)_LLRBP.txt" % (N, K, BP_iter_num))

    plot_data = np.loadtxt(plot_file, dtype=np.float32)
    x = plot_data[:, 0]
    y = plot_data[:, 1]
    if 0 == i:
        # label = format("BPDNN(%s,%s)iter=%s" % (N, K, BP_iter_num))
        label = format("(%s,%s)BPDNN(%s)" % (N, K, BP_iter_num))
    elif 1 == i:
        # label = format("advance_BPDNN(%s,%s)iter=%s" % (N, K, BP_iter_num + 20))
        label = format("(%s,%s)LLRBP(%s)" % (N, K, BP_iter_num))
    else:
        label = format("(%s,%s)LLRBP(%s)" % (N, K, BP_iter_num - 1))
    # elif 2 == i:
    #     # label = format("advance_BPDNN(%s,%s)iter=%s" % (N, K, BP_iter_num - 2))
    #     label = format("(%s,%s)BPDNN(%s)_CNN_BPDNN(%s)"
    #                    % (N, K, BP_iter_num // 2 - 1, BP_iter_num // 2 - 1))
    # else:
    #     label = format("(%s,%s)BP(%s)_CNN_BP(%s)" % (N, K, BP_iter_num // 2, BP_iter_num // 2))

    plt.semilogy(x, y, label=label, marker=marker, linestyle=linestyle)

plt.grid(True, which="both", ls="-")
plt.xlabel("SNR")
plt.ylabel("BER")
plt.legend()  # 启用图例（还可以设置图例的位置等等）
#plt.xlim(xmin=-0.5, xmax=3.5)
#plt.ylim(ymin=10e-3, ymax=0)
plt.show()