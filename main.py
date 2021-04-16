# coding=utf-8
#  #################################################################
#  Python code to reproduce our works on iterative BP-CNN.
#
#  Codes have been tested successfully on Python 3.4.5 with TensorFlow 1.1.0.
#
#  References:
#   Fei Liang, Cong Shen and Feng Wu, "An Iterative BP-CNN Architecture for Channel Decoding under Correlated Noise", IEEE JSTSP
#
#  Written by Fei Liang (lfbeyond@mail.ustc.edu.cn, liang.fei@outlook.com)
#  #################################################################


import sys
import Configrations
import numpy as np
import LinearBlkCodes as lbc
import Iterative_BP_CNN as ibd
import ConvNet
import DataIO

# address configurations
top_config = Configrations.TopConfig()
top_config.parse_cmd_line(sys.argv)

train_config = Configrations.TrainingConfig(top_config)
net_config = Configrations.NetConfig(top_config)

# (n,k)线性分组码，G:生成矩阵，H:校验矩阵， n 是经过生成矩阵之后的码长，k 是原来的码长
code = lbc.LDPC(top_config.N_code, top_config.K_code, top_config.file_G, top_config.file_H)
'''
更改码字的时候，需要配套修改的类有：
1. top_config中的N和K 、 matlab 中生成噪声的部分和 matlab 中生成生成矩阵和校验矩阵部分：
2. 是否需要启用训练的BP：BP_Decoder.py 中的self.use_train_bp_net ；是否需要启用conv net：net_config中的self.use_conv_net
    2.1 如果不需要使用 cnn,则将 top_config 中的 cnn_net_number 设为 0，同时将 BP_iter_nums_simu 设为只有一个元素。
3. 当启用新的码字时，先运行 train_bp_network 建立并训练对应的BP网络。
4. 如果恢复网络参数时,发生错误,提示网络不匹配,则有可能是batch_size不匹配,尝试删除已有的网络,并修改batch_size.
5. conv 和 bp 不能一起训练，bp 在获取session 的时候，会将 conv 的session获取过来，从而导致存储了双图。

-----------
从 https://www.uni-kl.de/channel-codes/channel-codes-database/ 下载下来的 alist 文件
0. 这个代码似乎只支持系统生成矩阵。
1. 首先经过 LDPC_alist 工程，得到chk mx 的稀疏矩阵中非零元素的坐标
2. 然后使用matlab中的 txt2H 得到对应的chk mx
3. 然后使用 matlab 中的 H2G 得到对应的 Gen mx，如果返回值的 valid=0，则将 此时得到的G和H做一个 exchHG，然后再做第三步。
5. 然后使用 find_x_y 得到 Gen mx 的稀疏矩阵的非零元素坐标

'''

batch_size = int(train_config.training_minibatch_size // np.size(train_config.SNR_set_gen_data))
batch_size //=2
BP_layers = 10  # BP的层数
train_epoch = 20000  # train epoch
use_weight_loss = True
if top_config.function == 'GenData':
    # 定义一个噪声生成器，读取的噪声是 [ 576 * 576 ]
    noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file) # top_config.cov_1_2_file = Noise/cov_1_2_corr_para_0.5.dat
    # generate training data

    # code is LDPC object，产生训练数据
    ibd.generate_noise_samples(code, top_config, net_config, train_config, top_config.BP_iter_nums_gen_data,
                                                  top_config.currently_trained_net_id, 'Training', noise_io, top_config.model_id, BP_layers)
    # generate test data，产生测试数据集
    ibd.generate_noise_samples(code, top_config, net_config, train_config, top_config.BP_iter_nums_gen_data,
                                                  top_config.currently_trained_net_id, 'Test', noise_io, top_config.model_id, BP_layers)
elif top_config.function == 'TrainConv':
    net_id = top_config.currently_trained_net_id
    # 定义一个卷积网络对象
    conv_net = ConvNet.ConvNet(net_config, train_config, net_id)
    # 开始训练网络

    conv_net.train_network(top_config.model_id, BP_layers)

elif top_config.function == 'TrainBP':
    # 训练 BP 网络
    ibd.train_bp_network(code, top_config, net_config, batch_size, BP_layers, train_epoch, use_weight_loss)

elif top_config.function == 'Simulation':
    # if top_config.analyze_res_noise:  # 分析残差噪声
    #     simutimes_for_anal_res_power = int(np.ceil(5e6 / float(top_config.K_code * batch_size)) * batch_size)
    #     ibd.analyze_residual_noise(code, top_config, net_config, simutimes_for_anal_res_power, batch_size, BP_layers)
    simutimes_range = np.array([np.ceil(8e6 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(8e7 / float(top_config.K_code * batch_size)) * batch_size], np.int32)  #　１ｅ７－１ｅ８
    ibd.simulation_colored_noise(code, top_config, net_config, simutimes_range, 1000, batch_size, BP_layers, train_epoch, use_weight_loss)
