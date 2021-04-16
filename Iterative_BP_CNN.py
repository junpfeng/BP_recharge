# This file contains some functions used to implement an iterative BP and denoising system for channel decoding.
# The denoising is implemented through CNN.
# The system architecture can be briefly denoted as BP-CNN-BP-CNN-BP...

import numpy as np
import datetime
import BP_Decoder
import tensorflow as tf
import ConvNet
import LinearBlkCodes as lbc
import DataIO


# empirical distribution ，经验分布？？ epdf: 经验概率密度函数
def stat_prob(x, prob):
    qstep = 0.01
    min_v = -10
    x = np.reshape(x, [1, np.size(x)])
    [hist, _] = np.histogram(x, np.int32(np.round(2*(-min_v) / qstep)),[min_v,-min_v])
    if np.size(prob) == 0:
        prob = hist
    else:
        prob = prob + hist

    return prob

# denoising and calculate LLR for next decoding
def denoising_and_calc_LLR_awgn(res_noise_power, y_receive, output_pre_decoder, net_in, net_out, sess):
    # estimate noise with cnn denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = sess.run(net_out, feed_dict={net_in: noise_before_cnn})
    # calculate the LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
    return LLR  # 返回新一轮BP_CNN的输入LLR

def calc_LLR_epdf(prob, s_mod_plus_res_noise):
    qstep = 0.01
    min_v = -10
    id = ((s_mod_plus_res_noise - 1 - min_v) / qstep).astype(np.int32)
    id[id < 0] = 0
    id[id > np.size(prob) - 1] = np.size(prob) - 1
    p0 = prob[id]
    id = ((s_mod_plus_res_noise + 1 - min_v) / qstep).astype(np.int32)
    id[id < 0] = 0
    id[id > np.size(prob) - 1] = np.size(prob) - 1
    p1 = prob[id]
    LLR = np.log(np.divide(p0 + 1e-7, p1 + 1e-7))  # 有可能是防止数值过小，所以加 1e-7
    return LLR

def denoising_and_calc_LLR_epdf(prob, y_receive, output_pre_decoder, net_in, net_out, sess):
    # estimate noise with cnn denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = sess.run(net_out, feed_dict={net_in: noise_before_cnn})
    # calculate the LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = calc_LLR_epdf(prob, s_mod_plus_res_noise)
    return LLR


# simulation
def simulation_colored_noise(linear_code, top_config, net_config, simutimes_range, target_err_bits_num, batch_size, BP_layers, train_epoch=25, use_weight_loss=False):
# target_err_bits_num: the simulation stops if the number of bit errors reaches the target.
# simutimes_range: [min_simutimes, max_simutimes]

    ## load configurations from top_config
    SNRset = top_config.eval_SNRs
    bp_iter_num = top_config.BP_iter_nums_simu
    noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file_simu, rng_seed=0)  # cov_1_2_file_simu 就是Noise文件夹下对应的噪声文件
    denoising_net_num = top_config.cnn_net_number
    model_id = top_config.model_id

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    K, N = np.shape(G_matrix)

    ## build BP decoding network
    if np.size(bp_iter_num) != denoising_net_num + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size, top_config, BP_layers, 0, use_weight_loss)
    # bp_decoder_after_cnn = BP_Decoder.BP_NetDecoder(H_matrix, batch_size, top_config, BP_layers, 1)
    # bp_decoder = bp_decoder_before_cnn  # default

    res_N = top_config.N_code
    res_K = top_config.K_code
    res_BP_layers = bp_decoder.BP_layers

    ## build denoising network
    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}
    # build network for each CNN denoiser,
    if net_config.use_conv_net:  # 如果使用 conv net 才加载
        for net_id in range(denoising_net_num):
            if top_config.same_model_all_nets and net_id > 0:
                conv_net[net_id] = conv_net[0]
                denoise_net_in[net_id] = denoise_net_in[0]
                denoise_net_out[net_id] = denoise_net_out[0]
            else:  # 默认进入
                # conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)  # 建立了一个残差噪声的神经网络对象
                conv_net[net_id] = ConvNet.ConvNet(net_config, top_config, net_id)  # 建立了一个残差噪声的神经网络对象
                denoise_net_in[net_id], denoise_net_out[net_id] = conv_net[net_id].build_network()  # 构建好对应的神经网络，返回的是网络的输入和输出
        # init gragh
        init = tf.global_variables_initializer()
        sess = tf.Session()
        print('Open a tf session!')
        sess.run(init)
        # restore denoising network
        for net_id in range(denoising_net_num):
            if top_config.same_model_all_nets and net_id > 0:
                break
            conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id+1)])  # 恢复之前训练好的网络。

    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(bp_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]

    if net_config.use_conv_net and bp_decoder.use_cnn_res_noise:
        ber_file = format('%s/bp_model/%s_%s/BP%s/BER(%d_%d)_BP(%s)_BPDNN%s-CNN-BPDNN%s'
                          % (net_config.model_folder, N, K, bp_decoder.BP_layers, N, K, bp_str, bp_decoder.BP_layers, bp_decoder.BP_layers))
        f_simulation_time = format('%s/bp_model/%s_%s/BP%s/simulation_time(%d_%d)_BP(%s)_BPDNN%s-CNN-BPDNN%s'
                                   % (net_config.model_folder, N, K, bp_decoder.BP_layers, N, K, bp_str, bp_decoder.BP_layers, bp_decoder.BP_layers))
    elif bp_decoder.use_train_bp_net or bp_decoder.train_bp_network:
        if use_weight_loss:
                ber_file = format('%s/bp_model/%s_%s/BP%s/BER(%d_%d)_BP(%s)_BP%s_epoch%s_weight_loss'
				  % (net_config.model_folder, N, K,bp_decoder.BP_layers, N, K, bp_str, bp_decoder.BP_layers, train_epoch))
                f_simulation_time = format('%s/bp_model/%s_%s/BP%s/simulation_time(%d_%d)_BP(%s)_BP%s_epch%s_weight_loss'
					   % (net_config.model_folder, N, K, bp_decoder.BP_layers, N, K, bp_str,bp_decoder.BP_layers, train_epoch))
        else:
                ber_file = format('%s/bp_model/%s_%s/BP%s/BER(%d_%d)_BP(%s)_BP%s_epoch%s'
				  % (net_config.model_folder, N, K,bp_decoder.BP_layers, N, K, bp_str, bp_decoder.BP_layers, train_epoch))
                f_simulation_time = format('%s/bp_model/%s_%s/BP%s/simulation_time(%d_%d)_BP(%s)_BP%s_epch%s'
					   % (net_config.model_folder, N, K, bp_decoder.BP_layers, N, K, bp_str,bp_decoder.BP_layers, train_epoch))

    else:
        ber_file = format('%s/bp_model/%s_%s/BP%s/BER(%d_%d)_BP(%s)_LLRBP%s'
                          % (net_config.model_folder, N, K, bp_decoder.BP_layers, N, K, bp_str, bp_decoder.BP_layers))
        f_simulation_time = format('%s/bp_model/%s_%s/BP%s/simulation_time(%d_%d)_BP(%s)_LLRBP%s'
                                    % (net_config.model_folder, N, K, bp_decoder.BP_layers, N, K, bp_str, bp_decoder.BP_layers))

    if top_config.corr_para != top_config.corr_para_simu:  # this means we are testing the model robustness to correlation level.
        ber_file = format('%s_SimuCorrPara%.2f' % (ber_file, top_config.corr_para_simu))
    if top_config.same_model_all_nets:
        ber_file = format('%s_SameModelAllNets' % ber_file)
    if top_config.update_llr_with_epdf:
        ber_file = format('%s_llrepdf' % ber_file)
    if denoising_net_num > 0:
        model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str)-1)]
        ber_file = format('%s_model%s' % (ber_file, model_id_str))
    if np.size(SNRset) == 1:
        ber_file = format('%s_%.1fdB' % (ber_file, SNRset[0]))

    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')
    simlation_time_file = format('%s.txt' % f_simulation_time)
    fout_simulation_time = open(simlation_time_file, 'wt')


    ## simulation starts
    start = datetime.datetime.now()
    total_simulation_times = 0
    residual_simulation_times = 0
    for SNR in SNRset:
        real_batch_size = batch_size
        # simulation part
        bit_errs_iter = np.zeros(denoising_net_num + 1, dtype=np.int32)
        actual_simutimes = 0
        rng = np.random.RandomState(1)  # 伪随机数种子
        noise_io.reset_noise_generator()  # reset随机数种子
        for ik in range(0, max_batches):  # 遍历max_batches 6667

            if ik == max_batches - 1 and residual_times != 0:  # 如果遍历结束，并且residual_times != 0 ，在这里默认是 == 0
                real_batch_size = residual_times
                residual_simulation_times = residual_simulation_times + 1
                fout_simulation_time.write('不足一个batch_size, 实际batch_size 是：' + str(real_batch_size) + '\n')
                print('不足一个batch_size, 实际batch_size 是：' + str(real_batch_size) + '\n')
            x_bits, u_coded_bits, s_mod, ch_noise, y_receive, LLR, ch_noise_sigma = lbc.encode_and_transmission(G_matrix, SNR, real_batch_size, noise_io, rng)  #
            # ------------------------------------------------------------
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0))
            if ik % 1000 == 0:
                print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
                print('Practical EbN0: %.2f' % practical_snr)

            for iter in range(0, denoising_net_num ):   # denoising_net_num == 1
                # if 0 == iter:
                #     bp_decoder = bp_decoder_before_cnn
                # else:
                #     bp_decoder = bp_decoder_after_cnn
                # BP decoding，第二个参数bp_iter_num 失效的，因为迭代次数是由前面的变量 BP_layers 决定的
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])  # BP译码传输的本来是LLR，返回的则是对应译码的码字
                # ！！！当iter==0，误比特率记录的是BP的误比特率，当iter==1，记录的是BP-CNN-BP的误比特率。
                # 首先判断是否使用 conv net
                if net_config.use_conv_net and iter < denoising_net_num:  # denoising_net_num == 1，当iter==0，使用CNN进行噪声估计，当iter==0，不使用CNN，即单纯使用BP译码
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter].get_res_noise_pdf(model_id, res_N, res_K, res_BP_layers).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_epdf(prob, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)
                    elif bp_decoder.use_cnn_res_noise:  # 默认进入else
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id, SNRset, res_N, res_K, res_BP_layers).get(np.float32(SNR))  # 计算噪声功率，这个残差噪声功率貌似是存储在文件中读取的
                        LLR = denoising_and_calc_LLR_awgn(res_noise_power, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)  # 使用神经网络译码进行噪声估计，并得到新一轮BP的LLR输入
                    else:
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id, SNRset, res_N, res_K, res_BP_layers).get(np.float32(SNR))  # 计算噪声功率，这个残差噪声功率貌似是存储在文件中读取的
                        LLR = denoising_and_calc_LLR_awgn(res_noise_power, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)  # 使用神经网络译码进行噪声估计，并得到新一轮BP的LLR输入
                        noise_after_cnn = y_receive - (u_BP_decoded * (-2) + 1)
                        # noise_after_cnn = sess.run(net_out, feed_dict={net_in: noise_before_cnn})
                        # calculate the LLR for next BP decoding
                        s_mod_plus_res_noise = y_receive - noise_after_cnn
                        LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
                output_x = linear_code.dec_src_bits(u_BP_decoded)  # 前k位是编码之前的信息位
                shape_x, shape_y = output_x.shape
                # for i in range(shape_x):
                #     if (np.any(output_x[i] - x_bits[i])):
                #         bit_errs_iter[iter] += 1
                bit_errs_iter[iter] += np.sum(output_x != x_bits)  # 统计比特不同的熟练（对应位比特不同记为1，然后累加计算有多少个不同比特位）
                pass
                # 同一个码字会记录两次误比特率，一次是只使用BP，还有一次是BP+CNN+BP。一般来说，经过BP+CNN+BP之后的误比特率要比只经过BP要好。

            actual_simutimes += real_batch_size
            if bit_errs_iter[denoising_net_num] >= target_err_bits_num and actual_simutimes >= min_simutimes:  # 当错误码元数或者仿真迭代次数达标
                break
        print('%d bits are simulated!, batch_size=%d' % (actual_simutimes * K, real_batch_size))

        total_simulation_times += actual_simutimes

        ber_iter = np.zeros(denoising_net_num+1, dtype=np.float64)
        fout_ber.write(str(SNR) + '\t')
        for iter in range(0, denoising_net_num+1):  # 1+1 = 2
            ber_iter[iter] = bit_errs_iter[iter] / float(K * actual_simutimes)
            fout_ber.write(str(ber_iter[iter]) + '\t' + str(bit_errs_iter[iter]) + '\t')
            print(ber_iter[iter])
        fout_ber.write('\n')
        # break

    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")

    fout_simulation_time.write(str(total_simulation_times) + '\t' + str((end-start).seconds))
    fout_simulation_time.close()

    if net_config.use_conv_net:
        sess.close()
    print('Close the tf session!')


def generate_noise_samples(linear_code, top_config, net_config, train_config, bp_iter_num, net_id_data_for, generate_data_for, noise_io, model_id, BP_layers=20):
    """
    :param linear_code: LDPC码对象
    :param top_config: 
    :param net_config: 
    :param train_config: 
    :param bp_iter_num: 
    :param net_id_data_for: 
    :param generate_data_for: 
    :param noise_io: 
    :param model_id: 
    :return: 
    """
# net_id_data_for: the id of the CNN network this function generates data for. Start from zero.
# model_id is to designate the specific model folder

    # 生成矩阵和校验矩阵
    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix

    SNRset_for_generate_training_data = train_config.SNR_set_gen_data
    if generate_data_for == 'Training':
        batch_size_each_SNR = int(train_config.training_minibatch_size // np.size(train_config.SNR_set_gen_data))
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
    elif generate_data_for == 'Test':
        batch_size_each_SNR = int(train_config.test_minibatch_size // np.size(train_config.SNR_set_gen_data))
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
    else:
        print('Invalid objective of data generation!')
        exit(0)


    # build BP decoding network
    if np.size(bp_iter_num) != net_id_data_for + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size_each_SNR, top_config, BP_layers)

    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}
    for net_id in range(net_id_data_for):
        # conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)
        conv_net[net_id] = ConvNet.ConvNet(net_config, train_config, net_id)
        denoise_net_in[net_id], denoise_net_out[net_id] = conv_net[net_id].build_network()

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # restore cnn networks before the target CNN
    for net_id in range(net_id_data_for):
        conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id+1)])

    start = datetime.datetime.now()

    if generate_data_for == 'Training':
        train_feature_path = train_config.training_feature_folder + format("BP%s/" % bp_decoder.BP_layers) \
                             + train_config.training_feature_file
        fout_est_noise = open(train_feature_path, 'wb')

        train_label_path = train_config.training_label_folder + format("BP%s/" % bp_decoder.BP_layers) \
                             + train_config.training_label_file
        fout_real_noise = open(train_label_path, 'wb')
        # fout_est_noise = open(train_config.training_feature_file, 'wb')
        # fout_real_noise = open(train_config.training_label_file, 'wb')
    elif generate_data_for == 'Test':
        test_feature_path = train_config.test_feature_folder + format("BP%s/" % bp_decoder.BP_layers) \
                             + train_config.test_feature_file
        fout_est_noise = open(test_feature_path, 'wb')

        test_label_path = train_config.test_label_folder + format("BP%s/" % bp_decoder.BP_layers) \
                          + train_config.test_label_file
        fout_real_noise = open(test_label_path, 'wb')
        # fout_est_noise = open(train_config.test_feature_file, 'wb')
        # fout_real_noise = open(train_config.test_label_file, 'wb')
    else:
        print('Invalid objective of data generation!')
        exit(0)

    # generating data，cnn网络的数据集产生：输入是经过BP译码输出数据noise_before_cnn，输出是实际的信道噪声：channel_noise
    for ik in range(0, total_batches):  # number of batches
        for SNR in SNRset_for_generate_training_data:
            x_bits, _, _, channel_noise, y_receive, LLR,_ = lbc.encode_and_transmission(G_matrix, SNR, batch_size_each_SNR, noise_io)
            # x_bits, 1 - u_coded_bits, s_mod, ch_noise, y_receive, LLR, ch_noise_sigma
            for iter in range(0, net_id_data_for + 1):
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])

                if iter != net_id_data_for:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_epdf(prob, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)
                    else:
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(res_noise_power, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)

            # reconstruct noise
            noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
            noise_before_cnn = noise_before_cnn.astype(np.float32)
            noise_before_cnn.tofile(fout_est_noise)  # write features to file
            channel_noise.tofile(fout_real_noise)  # write labels to file

    fout_real_noise.close()
    fout_est_noise.close()

    sess.close()
    end = datetime.datetime.now()

    print("Time: %ds" % (end - start).seconds)
    print("end")


## calculate the resdual noise power or its empirical distribution，分析残差噪声的经验分布或者噪声功率！！！！
def analyze_residual_noise(linear_code, top_config, net_config, simutimes, batch_size,BP_layers):

    ## load some configurations from top_config
    net_id_tested = top_config.currently_trained_net_id
    model_id = top_config.model_id
    bp_iter_num = top_config.BP_iter_nums_gen_data[0:(net_id_tested + 1)]
    noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file)
    SNRset = top_config.eval_SNRs

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    _, N = np.shape(G_matrix)

    max_batches, residual_times = np.array(divmod(simutimes, batch_size), np.int32)
    print('Real simutimes: %d' % simutimes)
    if residual_times != 0:
        max_batches += 1

    # build BP decoding network
    if np.size(bp_iter_num) != net_id_tested + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size, top_config, BP_layers)

    # build denoising network
    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}

    # build network for each CNN denoiser,
    for net_id in range(net_id_tested+1):
        # conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)
        conv_net[net_id] = ConvNet.ConvNet(net_config, top_config, net_id)
        denoise_net_in[net_id], denoise_net_out[net_id] = conv_net[net_id].build_network()

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # restore denoising network
    for net_id in range(net_id_tested + 1):
        conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id+1)])

    model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
    model_id_str = model_id_str[1:(len(model_id_str) - 1)]
    loss_file_name = format("%s/bp_model/%s_%s/BP%s/%s_%s_residual_noise_property_netid%d_model%s.txt"
                            % (net_config.residual_noise_property_folder, N, _, bp_decoder.BP_layers, N, _, net_id_tested, model_id_str))
    fout_loss = open(loss_file_name, 'wt')

    start = datetime.datetime.now()
    for SNR in SNRset:  # 0 0.5 1 1.5 2 2.5 3
        noise_io.reset_noise_generator()
        real_batch_size = batch_size
        # simulation part
        loss = 0.0
        prob = np.ones(0)
        for ik in range(0, max_batches):  # max_batches 3
            print("Batch id: %d" % ik)
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
            x_bits, _, s_mod, channel_noise, y_receive, LLR, _ = lbc.encode_and_transmission(G_matrix, SNR, real_batch_size, noise_io)
            # x_bits 随机生成的发送端码元，u_coded_bits 对x_bits做纠错编码后的码元，s_mod 对u_coded_bits做BPSK调制后的码元，ch_noise 信道噪声，y_recive 接收端接收到的信号，LLR 对数似然比
            for iter in range(0, net_id_tested+1):
                # BP decoding，astype：类型转为float32
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])  # u_BP_decoded 就是解码所得, bp_iter_num[iter] 表示bp迭代次数
                noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)  # 转为[-1,1]
                noise_after_cnn = sess.run(denoise_net_out[iter], feed_dict={denoise_net_in[iter]: noise_before_cnn})  #  cnn 计算噪声 n~
                s_mod_plus_res_noise = y_receive - noise_after_cnn  # 接收信号减去 cnn 的噪声 y~，
                if iter < net_id_tested:  # calculate the LLR for next BP decoding
                    if top_config.update_llr_with_epdf:  # 这里就决定了 经过 cnn 输出的残差噪声以什么形式转入下一轮的迭代（经验分布或者重新计算）默认是经验分布
                        prob_tmp = conv_net[iter].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = calc_LLR_epdf(prob_tmp, s_mod_plus_res_noise)
                    else:
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id).get(np.float32(SNR))
                        LLR = s_mod_plus_res_noise * 2.0 / res_noise_power  # 计算新一轮的BP输入
            if top_config.update_llr_with_epdf:
                prob = stat_prob(s_mod_plus_res_noise - s_mod, prob)  # s_mod 是发送端的悉尼号
            else:  # 累加实际值和网络输出值之间的均方误差
                loss += np.sum(np.mean(np.square(s_mod_plus_res_noise-s_mod), 1))  # 求的是新一轮迭代的输入噪声的平均功率e
                #   axis=1 表示计算[[a,b],[c,d]] [(a+b)/2, (c+d)/2]，损失函数是均方误差

        # each SNR 对应的CNN的loss。
        if top_config.update_llr_with_epdf:
            fout_loss.write(str(SNR) + '\t')
            for i in range(np.size(prob)):
                fout_loss.write(str(prob[i]) + '\t')
            fout_loss.write('\n')
        else:
            loss /= np.double(simutimes * 16)  # 猜测 simutimes = 5000*3, 其中5000是一次测试5000行马元，3是测试三次即 max_batches = 3
            fout_loss.write(str(SNR) + '\t' + str(loss) + '\n')  # residual_noise_property_netid0_model0.txt 里面存储的是损失值，很明显损失值是在递减的，说明训练有效果

    fout_loss.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()

    #  训练BP网络
    # simulation
def train_bp_network(linear_code, top_config, net_config, batch_size, BP_layers=20, train_epoch=25, use_weight_loss=False):
    # target_err_bits_num: the simulation stops if the number of bit errors reaches the target.
    # simutimes_range: [min_simutimes, max_simutimes]

    ## load configurations from top_config
    SNRset = top_config.eval_SNRs
    bp_iter_num = top_config.BP_iter_nums_simu
    # noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file_simu, rng_seed=0)
    # ch_noise_normalize = noise_io.generate_noise(batch_size)  # 生成均值为0，方差为1的高斯随机噪声矩阵（相干性设为0）
    ch_noise_normalize = 0
    denoising_net_num = top_config.cnn_net_number
    # model_id = top_config.model_id

    # G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    # K, N = np.shape(G_matrix)

    ## build BP decoding network
    if np.size(bp_iter_num) != denoising_net_num + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size, top_config, BP_layers,0, use_weight_loss)

    ## train bp_network
    start = datetime.datetime.now()
    bp_decoder.train_decode_network(bp_iter_num[0], SNRset, batch_size, ch_noise_normalize, linear_code, train_epoch)  # BP译码传输的本来是LLR，返回的则是对应译码的码字
    end = datetime.datetime.now()
    print('Time: %ds' % (end - start).seconds)
    print("end\n")
