# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 仿真时间统一换算到 100 w个码字
# iter=25：BPDNN-BDCNN:H(576,432) : 767,892
# iter=50: BPDNN-LLRBP:H(576, 432):767, 1293

name_list = ['DNN', 'LLRBBP50']
name_list = ['BPDNN20', 'advance_BPDNN20', 'LLRBBP50']
# name_list = ['BPDNN25', 'LLRBP50']
num_list = [614, 759, 1293]
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list, width=0.2)
plt.ylabel("time/s")
# plt.title('LDPC(576, 432) decode simulation time')

plt.legend()  # 启用图例（还可以设置图例的位置等等）
plt.show()