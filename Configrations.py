# coding=utf-8
import numpy as np

## TopConfig defines some top configurations. Other configurations are set based on TopConfig. TopConfig 是相当于配置类的基类
class TopConfig:  # 也就是 生成数据、训练和仿真的 参数和配置信息
    """
    需要修改的参数有：
    1.  N_code & K_code
    2.  corr_para  # 相干系数，为0时，非相干（同时matlab需要配套修改）
    3.  BP_iter_nums_simu  # BP和CNN的迭代次数
    """
    def __init__(self):
        # select functions to be executed, including generating data(GenData), training(Train), and simulation(Simulation)
        self.function = 'Train'

        # code 码的信息
        self.N_code = 576
        self.K_code = 432
        # self.N_code = 576
        # self.K_code = 480
        # self.N_code = 240
        # self.K_code = 120
        # self.N_code = 6
        # self.K_code = 3
        # self.N_code = 16
        # self.K_code = 8
        # self.N_code = 96
        # self.K_code = 48
        # self.K_code = 45
        # self.N_code = 63
        # self.N_code = 128
        # self.K_code = 64
        # self.N_code = 32
        # self.K_code = 16
        self.file_G = format('./LDPC_matrix/LDPC_gen_mat_%d_%d.txt' % (self.N_code, self.K_code))
        self.file_H = format('./LDPC_matrix/LDPC_chk_mat_%d_%d.txt' % (self.N_code, self.K_code))

        # noise information
        self.blk_len = self.N_code
        # self.corr_para = 0.5  # correlation parameters of the colored noise 相关系数
        self.corr_para = 0.0  # correlation parameters of the colored noise 相关系数
        self.corr_para_simu = self.corr_para  # correlation parameters for simulation. this should be equal to corr_para. If not, it is used to test the model robustness.
        self.cov_1_2_file = format('./Noise/%s_%s/cov_1_2_corr_para%.2f.dat' % (self.N_code, self.K_code, self.corr_para))
        self.cov_1_2_file_simu = self.cov_1_2_file

        # BP decoding
        self.BP_iter_nums_gen_data = np.array([5])     # the number of BP iterations
        # self.BP_iter_nums_simu = np.array([5,5])  # [5,5]  % BP（5）-CNN-BP（5）
        self.BP_iter_nums_simu = np.array([5, 5])  # np.array([5, 5]) # 配合下面cnn_net_number # [5,5]  % BP（5）-CNN-BP（5）

        # cnn config
        self.currently_trained_net_id = 0  # denote the cnn denoiser which is in training currently

        # 表示整个网络有多少个小的独立BP网络拼接而成 0表示1个，1表示2个
        self.cnn_net_number = 1  # 1  # 当需要使用cnn时，设为1 ，the number of cnn denoisers in final simulation

        self.layer_num = 4  # the number of cnn layers
        self.filter_sizes = np.array([9, 3, 3, 15])  # the convolutional filter size. The length of this list should be equal to the layer number
        self.feature_map_nums = np.array([64, 32, 16, 1])  # the last element must be 1
        self.restore_network_from_file = False  # whether to restore previous saved network for training
        self.model_id = np.array([0])  # differentiate models trained with the same configurations. Its length should be equal to cnn_net_number. model_id[i] denotes the index of
        #  the ith network in the BP-CNN-BP-CNN-... structure.

        # Trianing
        self.normality_test_enabled = True
        self.normality_lambda = 1

        if self.N_code == 16 and self.K_code == 8:
            self.SNR_set_gen_data = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=np.float32)  # 这个修改和 eval_SNRs 是匹配的
            self.eval_SNRs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], np.float32)
            # 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6,
            # 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13
            # self.SNR_set_gen_data = np.array([1,2,3,4,5,6,7,8,9,10]
            #                                   # 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13,
            #                                   # 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20]
            #                                 , dtype=np.float32)  # 这个修改和 eval_SNRs 是匹配的
            # self.eval_SNRs = np.array([1,2,3,4,5,6,7,8,9,10]
                                      # [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
            #                            # 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13,
            #                            # 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20]
            #                           , np.float32)
        else:
            # self.SNR_set_gen_data = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
            #                                  , dtype=np.float32)  # 这个修改和 eval_SNRs 是匹配的
            # self.eval_SNRs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
            #                           , np.float32)
            self.SNR_set_gen_data = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]
                                             , dtype=np.float32)  # 这个修改和 eval_SNRs 是匹配的
            self.eval_SNRs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]
                                      , np.float32)
            # self.SNR_set_gen_data = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=np.float32)  # 这个修改和 eval_SNRs 是匹配的
            # self.eval_SNRs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], np.float32)  # 这个修改和 SNR_set_gen_data 是匹配的

        self.SNR_set_origin_size = 7  # 这个是默认的信噪比集合的长度 7
        self.same_model_all_nets = False  # denote whether the same model parameters for all denoising networks. If true and cnn_net_number > 1, we are testing the performance
        #  of iteration between a BP and a denoising network.
        self.analyze_res_noise = True
        self.update_llr_with_epdf = False  # whether to update the initial LLR of the next BP decoding with the empirical distribution. Otherwise, the LLR is updated by
        # viewing the residual noise follows a Gaussian distritbution


    def parse_cmd_line(self, argv):
        """
        命令行参数的解析
        :param argv: 
        :return: 
        """
        if len(argv) == 1:
            return
        id = 1
        while id < len(argv):
            if argv[id]=='-Func':
                self.function = argv[id+1]
                print('Function is set to %s' % argv[id+1])
            # noise information
            elif argv[id]=='-CorrPara':
                self.corr_para = float(argv[id+1])
                self.cov_1_2_file = format('./Noise/cov_1_2_corr_para%.2f.dat' % self.corr_para)
                print('Corr para is set to %.2f' % self.corr_para)
            # Simulation
            elif argv[id]=='-UpdateLLR_Epdf':
                self.update_llr_with_epdf = (argv[id+1] == 'True')
            elif argv[id]=='-EvalSNR':
                self.eval_SNRs = np.fromstring(argv[id+1], np.float32, sep=' ')
                print('eval_SNRs is set to %s' % np.array2string(self.eval_SNRs))
            elif argv[id]=='-AnalResNoise':
                self.analyze_res_noise = (argv[id+1] == 'True')
                print('analyze_res_noise is set to %s' % str(self.analyze_res_noise))
            elif argv[id]=='-SimuCorrPara':
                self.corr_para_simu = float(argv[id+1])
                self.cov_1_2_file_simu = format('./Noise/cov_1_2_corr_para%.2f.dat' % self.corr_para_simu)
                print('Corr para for simulation is set to %.2f' % self.corr_para_simu)
            elif argv[id]=='-SameModelAllNets':
                self.same_model_all_nets = (argv[id+1]=='True')
                print('same_model_all_nets is set to %s' % str(self.same_model_all_nets))
            # BP decoding
            elif argv[id]=='-BP_IterForGenData':
                self.BP_iter_nums_gen_data = np.fromstring(argv[id+1], np.int32, sep=' ')
                print('BP iter for gen data is set to: %s' % np.array2string(self.BP_iter_nums_gen_data))
            elif argv[id]=='-BP_IterForSimu':
                self.BP_iter_nums_simu = np.fromstring(argv[id+1], np.int32, sep=' ')
                print('BP iter for simulation is set to: %s' % np.array2string(self.BP_iter_nums_simu))
            # CNN config
            elif argv[id]=='-NetNumber':
                self.cnn_net_number = np.int32(argv[id+1])
            elif argv[id]=='-CNN_Layer':
                self.layer_num = np.int32(argv[id+1])
                print('CNN layer number is set to %d' % self.layer_num)
            elif argv[id]=='-FilterSize':
                self.filter_sizes = np.fromstring(argv[id+1], np.int32, sep=' ')
                print('Filter sizes are set to %s' % np.array2string(self.filter_sizes))
            elif argv[id]=='-FeatureMap':
                self.feature_map_nums = np.fromstring(argv[id + 1], np.int32, sep=' ')
                print('Feature map numbers are set to %s' % np.array2string(self.feature_map_nums))
            # training
            elif argv[id]=='-ModelId':
                self.model_id = np.fromstring(argv[id + 1], np.int32, sep=' ')
                print('Model id is set to %s' % (np.array2string(self.model_id)))
            elif argv[id]=='-NormTest':
                self.normality_test_enabled = (argv[id+1] == 'True')
                print('Normality test: %s' % str(self.normality_test_enabled))
            elif argv[id]=='-NormLambda':
                self.normality_lambda = np.float32(argv[id+1])
                print('Normality lambda is set to %f' % self.normality_lambda)
            elif argv[id]=='-SNR_GenData':
                self.SNR_set_gen_data = np.fromstring(argv[id+1], dtype=np.float32, sep=' ')
                print('SNR set for generating data is set to %s.' % np.array2string(self.SNR_set_gen_data))
            else:
                print('Invalid parameter %s!' % argv[id])
                exit(0)
            id = id + 2




# class for network configurations
class NetConfig:
    """
    神经网络的配置信息和参数设置
    """
    def __init__(self, top_config):
        # network parameters
        if top_config.restore_network_from_file:
            self.restore_layers = top_config.layer_num
        else:
            self.restore_layers = 0
        self.save_layers = top_config.layer_num
        self.total_layers = top_config.layer_num  # the input layer is not included but the output layer is included
        self.feature_length = top_config.blk_len
        self.label_length = top_config.blk_len
        self.node_num_each_layer = np.ones(top_config.layer_num, dtype=np.int32) * self.feature_length  # the length of this array must be the same with the number of cnn layers

        # conv net parameters
        self.filter_sizes = top_config.filter_sizes
        self.feature_map_nums = top_config.feature_map_nums
        self.layer_num = top_config.layer_num

        self.model_folder = "./model"
        self.residual_noise_property_folder = self.model_folder
        self.use_conv_net = False  # True


class TrainingConfig:
    """
    训练过程的配置信息
    """
    def __init__(self, top_config):  # 相当于继承于top_config

        # cov^(1/2) file
        self.corr_para = top_config.corr_para

        self.currently_trained_net_id = top_config.currently_trained_net_id

        # training data information
        self.training_sample_num = 1999200 // top_config.SNR_set_origin_size * np.size(top_config.SNR_set_gen_data) # 1999200    # the number of training samples. It should be a multiple of training_minibatch_size
        # training parameters
        self.epoch_num = 200000   # 200000  # the number of training iterations.本来是训练200000轮，为了提高速度，改成1轮
        self.training_minibatch_size = 2800 // top_config.SNR_set_origin_size * np.size(top_config.SNR_set_gen_data)
        # 1400  # one mini-batch contains equal amount of data generated under different CSNR.
        self.SNR_set_gen_data = top_config.SNR_set_gen_data
        # the data in the feature file is the network input.
        # the data in the label file is the ground truth.
        self.training_feature_folder = format("./TrainingData/%s_%s/" % (top_config.N_code, top_config.K_code))
        self.training_feature_file = format("EstNoise_before_cnn%d.dat" % (self.currently_trained_net_id))
        self.training_label_folder = format("./TrainingData/%s_%s/" % (top_config.N_code, top_config.K_code))
        self.training_label_file = format("RealNoise.dat")

        # test data information
        self.test_sample_num = 42000 // top_config.SNR_set_origin_size * np.size(top_config.SNR_set_gen_data) # 原来是 105000 # it should be a multiple of test_minibatch_size
        self.test_minibatch_size = 2800 // top_config.SNR_set_origin_size * np.size(top_config.SNR_set_gen_data)
        # 原来是 3500，为了配合训练BP，改成和上面training_minibatch_size一样的 1400
        self.test_feature_folder = format("./TestData/%s_%s/" % (top_config.N_code, top_config.K_code))
        self.test_feature_file = format("EstNoise_before_cnn%d.dat" % (self.currently_trained_net_id))
        self.test_label_folder = format("./TestData/%s_%s/" % (top_config.N_code, top_config.K_code))
        self.test_label_file = format("RealNoise.dat")

        # normality test
        self.normality_test_enabled = top_config.normality_test_enabled
        self.normality_lambda = top_config.normality_lambda

        ## parameter check
        if self.test_sample_num % self.test_minibatch_size != 0:
            print('Total_test_samples must be a multiple of test_minibatch_size!')
            exit(0)
        if self.training_sample_num % self.training_minibatch_size != 0:
            print('Total_training_samples must be a multiple of training_minibatch_size!')
            exit(0)
        if self.training_minibatch_size % np.size(self.SNR_set_gen_data)!=0 or self.test_minibatch_size % np.size(self.SNR_set_gen_data)!=0:
            print('A batch of training or test data should contains equal amount of data under different CSNRs!')
            exit(0)

        # code feature
        self.N_code = top_config.N_code
        self.K_code = top_config.K_code
