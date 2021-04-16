import numpy as np
import tensorflow as tf

# this file defines classes for data io
class TrainingDataIO:
    def __init__(self, feature_filename, label_filename, total_trainig_samples, feature_length, label_length):
        print("Construct the data IO class for training!\n")
        self.fin_label = open(label_filename, "rb")
        # self.fin_feature 是文件对象
        self.fin_feature = open(feature_filename, "rb")
        self.total_trainig_samples = total_trainig_samples
        self.feature_length = feature_length
        self.label_length = label_length

    def __del__(self):
        print("Delete the data IO class!\n")
        self.fin_feature.close()
        self.fin_label.close()

    def load_next_mini_batch(self, mini_batch_size, factor_of_start_pos=1):
        """
        采用的是 随机-批量处理 算法
        :param mini_batch_size: 1400 
        :param factor_of_start_pos: 1
        :return: 
        """
        # get 数据集的 输入和输出 非常重要。
        # the function is to load the next batch where the datas in the batch are from a continuous memory block
        # the start position for reading data must be a multiple of factor_of_start_pos
        remain_samples = mini_batch_size
        # 产生一个正态分布的随机数，这个随机数比 self.total_trainig_samples 小，其实就是选择训练数据中的第 sample_id 个数据。
        sample_id = np.random.randint(self.total_trainig_samples)   # output a single value which is less than total_trainig_samples
        features = np.zeros((0))
        labels = np.zeros((0))
        if mini_batch_size > self.total_trainig_samples:  # 如果批量数都大于这个样本总量了，那么数据量就不够了
            print("Mini batch size should not be larger than total sample size!\n")

        # 文件操作，seek,移动文件指针，0为初始值，向后移动 (self.feature_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos) 个字节（bytes)
        # 下面两个操作，就是将文件指针指向了第 sample_id 个数据的开始。
        self.fin_feature.seek((self.feature_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos), 0)  # float32 = 4 bytes = 32 bits
        self.fin_label.seek((self.label_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos), 0)

        while 1:
            #TODO 下面读取数据之所以会这么复杂，是为了对付一种情况：从 sample_id 向后，数据量不足 mini_batch_size。不足，就从头开始再找点数据。
            #TODO 不得不说，这种处理非常有趣。
            # self.fin_feature是上面转移的文件指针，np.float32是读取文件之后转换为的数据类型，self.feature_length*remain_samples 是读取的字节数。返回读取的数据
            new_feature = np.fromfile(self.fin_feature, np.float32, self.feature_length * remain_samples)
            new_label = np.fromfile(self.fin_label, np.float32, self.label_length * remain_samples)
            # 将 feature 和 new_feature 两个二维张量拼接起来
            features = np.concatenate((features, new_feature))
            labels = np.concatenate((labels, new_label))


            # 求剩余的样本数量
            remain_samples -= len(new_feature) // self.feature_length
            # TODO 如果remain_samples != 0,说明，数据没有取够。
            # 数据全部训练完毕则退出
            if remain_samples == 0:
                break
            # 还原文件指针
            self.fin_feature.seek(0, 0)
            self.fin_label.seek(0, 0)
        # 将取出的数据进行尺寸转换，然后返回。
        features = features.reshape((mini_batch_size, self.feature_length))
        labels = labels.reshape((mini_batch_size, self.label_length))
        return features, labels


class TestDataIO:
    def __init__(self, feature_filename, label_filename, test_sample_num, feature_length, label_length):
        self.fin_label = open(label_filename, "rb")
        self.fin_feature = open(feature_filename, "rb")
        self.test_sample_num = test_sample_num
        self.feature_length = feature_length
        self.label_length = label_length
        self.all_features = np.zeros(0)
        self.all_labels = np.zeros(0)
        self.data_position = 0

    def __del__(self):
        self.fin_feature.close()
        self.fin_label.close()

    def seek_file_to_zero(self):  # reset the file pointer to the start of the file
        self.fin_feature.seek(0, 0)
        self.fin_label.seek(0, 0)

    def load_batch_for_test(self, batch_size):
        if batch_size > self.test_sample_num:
            print("Batch size should not be larger than total sample size!\n")
        if np.size(self.all_features) == 0:
            self.all_features = np.fromfile(self.fin_feature, np.float32, self.feature_length * self.test_sample_num)
            self.all_labels = np.fromfile(self.fin_label, np.float32, self.label_length * self.test_sample_num)
            self.all_features = np.reshape(self.all_features, [self.test_sample_num, self.feature_length])  # 出错的原因：self.all_feature < self.test_sample_num * self.feature_length
            self.all_labels = np.reshape(self.all_labels, [self.test_sample_num, self.label_length])

        features = self.all_features[self.data_position:(self.data_position + batch_size), :]
        labels = self.all_labels[self.data_position:(self.data_position + batch_size), :]
        self.data_position += batch_size
        if self.data_position >= self.test_sample_num:
            self.data_position = 0
        return features, labels


class NoiseIO:
    """
    read_from_file: False
    noise_file: None
    """
    def __init__(self, blk_len, read_from_file, noise_file, cov_1_2_mat_file_gen_noise, rng_seed=None):  # 第三个参数 noise_file 是扩展为自定义的噪声文件用，暂时没有使用。
        self.read_from_file = read_from_file  # False，不从外部文件导入
        self.blk_len = blk_len  # blk是线性分组码的简称，len是码长
        self.rng_seed = rng_seed  # 随机数种子
        if read_from_file:
            self.fin_noise = open(noise_file, 'rb')
        else:
            self.rng = np.random.RandomState(rng_seed)  # 定义一个随机数对象 rng.rand(n)可以产生n个标准正态分布随机数
#            fin_cov_file = open(cov_1_2_mat_file_gen_noise, 'rb')  # cov_1_2_mat_file_gen_noise = Noise/cov_1_2_corr_para_0.5.dat
#            cov_1_2_mat = np.fromfile(fin_cov_file, np.float32, blk_len*blk_len)  # np.fromfile 会读取 fin_cov_file 中的二进制数据，读取blk_len*ble_len个，保存为float32类型
#            cov_1_2_mat = np.reshape(cov_1_2_mat, [blk_len, blk_len])  # 将读取的数据格式转为 [ble_len,ble_len]的单位矩阵。
            # 关闭文件管理器
#            fin_cov_file.close()
            # output parts of the correlation function for check
#            self.cov_func = np.matmul(cov_1_2_mat, cov_1_2_mat)  # 矩阵乘法，计算协方差
            self.cov_func = np.eye(self.blk_len)
            print('Correlation function of channel noise: ')
            print(self.cov_func[0,0:10])
            # 放置一个 placeholder -- 噪声矩阵
            # 建立一个独立的图 g1
            # self.g1 = tf.Graph()
            # with self.g1.as_default():
            #     self.awgn_noise = tf.placeholder(dtype=tf.float32, shape=[None, blk_len])
            #     self.noise_tf = tf.matmul(self.awgn_noise, cov_1_2_mat)  # 这边建立了一个产生噪声的网络
            # 开启一个会话
            # self.sess = tf.Session(graph=self.g1)

    def __del__(self):
        if self.read_from_file:
            self.fin_noise.close()
        else:
            # self.sess.close()
            pass

    def reset_noise_generator(self): # this function resets the file pointer or the rng generator to generate the same noise data
        if self.read_from_file:
            self.fin_noise.seek(0, 0)
        else:
            self.rng = np.random.RandomState(self.rng_seed)


    def generate_noise(self, batch_size):
        if self.read_from_file:  # 是指第三方提供的噪声文件
            noise = np.fromfile(self.fin_noise, np.float32, batch_size * self.blk_len)
            noise = np.reshape(noise, [batch_size, self.blk_len])
        else:   # 使用 matlab 提供的噪声文件
            noise_awgn = self.rng.randn(batch_size, self.blk_len)  # 产生均值为0，方差为1的随机数矩阵
            noise_awgn = noise_awgn.astype(np.float32)
            # 感觉没必要用图，就直接计算结果
            noise = np.matmul(noise_awgn, self.cov_func)
            # with tf.Session(graph=self.g1) as sess:
            #     noise = sess.run(self.noise_tf, feed_dict={self.awgn_noise: noise_awgn})  # __init__ 里面建立了一个产生相关噪声的网络，然后输出相关噪声
        return noise
