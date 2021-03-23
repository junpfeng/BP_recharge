import tensorflow as tf
import numpy as np
import os
import LinearBlkCodes as lbc
import DataIO
import Configrations

top_config = Configrations.TopConfig()


class GetMatrixForBPNet:
    # this class is to calculate the matrices used to perform BP process with matrix operation
    # test_H即校验矩阵，loc_nzero_row是校验矩阵中非零元素的坐标（横坐标和纵坐标分别存储）
    def  __init__(self, test_H, loc_nzero_row):
        print("Construct the Matrics H class!\n")
        self.H = test_H
        self.m, self.n = np.shape(test_H)  # 校验矩阵是 144 行，576列！！！
        self.H_sum_line = np.sum(self.H, axis=0)  # 将校验矩阵每列相加(表示每列非零元素数量)，由（144, 576)变成 （1，576）,0-431元素都是4，432-455元素是3，456-575元素是2，总的1元素数量2040
        self.H_sum_row = np.sum(self.H, axis=1)  # 同上，每行相加(表示每列非零元素数量)，由(144, 576)变成（144，1）,其中72-95元素是15，其余元素都是14，于是，总的1的数量 =2040
        self.loc_nzero_row = loc_nzero_row
        self.num_all_edges = np.size(self.loc_nzero_row[1, :])  # 校验矩阵中所有1的元素数量是2040
        #  各种统计数据????
        self.loc_nzero1 = self.loc_nzero_row[1, :] * self.n + self.loc_nzero_row[0, :]  # 这种计算感觉是某种编码
        self.loc_nzero2 = np.sort(self.loc_nzero1)  # 进行排序
        self.loc_nzero_line = np.append([np.mod(self.loc_nzero2, self.n)], [self.loc_nzero2 // self.n], axis=0)  # 转为两行，第一行余数，第二行整除商，这个正好是edge的坐标（竖向排序）
        self.loc_nzero4 = self.loc_nzero_line[0, :] * self.n + self.loc_nzero_line[1, :]  # 余数 * 576 + 对应的商 正好是edge对应的位置（横向）
        self.loc_nzero5 = np.sort(self.loc_nzero4)
        # loc_nzero_line 内的是按找竖向顺序排列的非零元素的坐标
        # loc_nzero4 是非零元素的位置（比如（0，0）就是第0个非零元素，（0，1）就是第一个），元素排放顺序则是和loc_nzero_line 一致
        # loc_nzero5 则是单纯 loc_nzero4的排序


    ##########################################################################################################
    def get_Matrix_VC(self):
        H_x_to_xe0 = np.zeros([self.num_all_edges, self.n], np.float32)  # (2040, 576)
        H_sum_by_V_to_C = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)  # (2040, 2040)
        H_xe_last_to_y = np.zeros([self.n, self.num_all_edges], dtype=np.float32)  # (576, 2040)
        Map_row_to_line = np.zeros([self.num_all_edges, 1])  # (2040,1)

        for i in range(0, self.num_all_edges):
            Map_row_to_line[i] = np.where(self.loc_nzero1 == self.loc_nzero2[i])  # 返回loc_nzerol 中等于 loc_nzero2[i]的元素的索引
            # !!! Map_row_to_line 记录了 loc_nzero_row 到 loc_nzero_line 之间的映射关系，不过使用的是数组形式的哈希表
            # Map_row_to_line 即横向edge到纵向edge的更新矩阵
        map_H_row_to_line = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, self.num_all_edges):
            map_H_row_to_line[i, int(Map_row_to_line[i])] = 1

        count = 0
        for i in range(0, self.n):
            temp = count + self.H_sum_line[i]
            H_sum_by_V_to_C[count:temp, count:temp] = 1
            H_xe_last_to_y[i, count:temp] = 1
            H_x_to_xe0[count:temp, i] = 1
            for j in range(0, self.H_sum_line[i]):
                H_sum_by_V_to_C[count + j, count + j] = 0
            count = count + self.H_sum_line[i]
        print("return Matrics V-C successfully!\n")
        return H_x_to_xe0, np.matmul(H_sum_by_V_to_C, map_H_row_to_line), np.matmul(H_xe_last_to_y, map_H_row_to_line)
        # H_x_to_xe0 是码字到输入层变量节点的转换矩阵
        # H_sum_by_V_to_C 是纵向（校验节点到变量节点）的更新矩阵，map_H_row_to_line 是将横向edge转为纵向edge
        # H_xe_last_to_y　是最后一层的转换矩阵

    ###################################################################################################
    def  get_Matrix_CV(self):  # 获取从 check node -> variable node 的变换s矩阵

        H_sum_by_C_to_V = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)  # 2040 * 2040 的矩阵
        Map_line_to_row = np.zeros([self.num_all_edges, 1])  # 2040 * 1 的矩阵
        for i in range(0, self.num_all_edges):
            Map_line_to_row[i] = np.where(self.loc_nzero4 == self.loc_nzero5[i])
        map_H_line_to_row = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, np.size(self.loc_nzero1)):
            map_H_line_to_row[i, int(Map_line_to_row[i])] = 1

        count = 0
        for i in range(0, self.m):
            temp = count + self.H_sum_row[i]
            H_sum_by_C_to_V[count:temp, count:temp] = 1
            for j in range(0, self.H_sum_row[i]):
                H_sum_by_C_to_V[count + j, count + j] = 0
            count = count + self.H_sum_row[i]
        print("return Matrics C-V successfully!\n")
        return np.matmul(H_sum_by_C_to_V, map_H_line_to_row)
        # H_sum_by_C_to_V 是横向更新的矩阵，map_H_line_to_row 是将纵向edge转为横向edge的矩阵


class BP_NetDecoder:
    def __init__(self, H, batch_size, top_config, BP_layers=20, placement=0):  # 校验矩阵，外部传入

        if top_config.function == 'GenData':
            self.train_bp_network = False
            self.use_train_bp_net = False  # True
            self.use_cnn_res_noise = False
        elif top_config.function == 'TrainBP':
            self.train_bp_network = True
            self.use_train_bp_net = False  # True
            self.use_cnn_res_noise = False
        elif top_config.function == 'Simulation':
            self.train_bp_network = False
            self.use_train_bp_net = True  # True
            self.use_cnn_res_noise = False
        else:
            self.train_bp_network = False
            self.use_train_bp_net = False  # True
            self.use_cnn_res_noise = True
        # 设置tf 初始化的模式
        self.initializer = tf.truncated_normal_initializer(mean=1, stddev=0.1)
        _, self.v_node_num = np.shape(H)  #  获取变量节点长度（即码元的长度 576）
        ii, jj = np.nonzero(H)  # 返回校验矩阵H中非零元素的索引，x轴依次返回给ii, y轴依次返回给jj，也就是说，非零元素的坐标表示位(ii,jj)
        loc_nzero_row = np.array([ii, jj])  # 将 ii 和 jj 组合起来了
        self.num_all_edges = np.size(loc_nzero_row[1, :])  # 获取非零元素的数量，同时也被称为edge，横坐标或者纵坐标的数量即edge数量
        gm1 = GetMatrixForBPNet(H[:, :], loc_nzero_row)  #
        self.H_sumC_to_V = gm1.get_Matrix_CV()  # 返回：C->V 的转换矩阵
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y = gm1.get_Matrix_VC()  # 返回：初始化的变量节点、V->C 的转换矩阵、输出层的转换矩阵

        # ------------------------------------
        self.batch_size = batch_size
        self.llr_placeholder = tf.placeholder(tf.float32, [batch_size, self.v_node_num], name="llr_placeholder")
        self.labels = tf.placeholder(tf.float32, [batch_size, self.v_node_num], name="label_placeholder")
        # -----------新增变量------------
        self.x_bit_placeholder = tf.placeholder(tf.int8, [batch_size, self.v_node_num])
        # ---------- BP 网络的参数 -------------
        self.V_to_C_params = {}
        self.C_to_V_params = {}
        self.C_to_V_params_img = {}
        self.BP_layers = BP_layers
        self.xe_v_sumc = {}

        # ---------- 构建稀疏转换 -------------
        # --------- 将参数放到数组中 ----------
        i, j = np.nonzero(self.H_sumC_to_V)
        indices = []
        for idx in range(len(i)):
            tmp = [0, 0]
            tmp[0] = i[idx]
            tmp[1] = j[idx]
            indices.append(tmp)
        for layer in range(self.BP_layers):
            values = tf.Variable(np.ones(len(i)), dtype=tf.float32, name=format("c_to_v_sparse_var_%d" % layer))
            self.C_to_V_params[layer] = tf.SparseTensor(indices=indices, values=values
                                                        , dense_shape=self.H_sumC_to_V.shape)
            values_img = tf.Variable(np.ones(len(i)), dtype=tf.float32, name=format("c_to_v_sparse_var_img_%d" % layer))
            self.C_to_V_params_img[layer] = tf.SparseTensor(indices=indices, values=values_img
                                                            , dense_shape=self.H_sumC_to_V.shape)

        i, j = np.nonzero(self.H_sumV_to_C)
        indices = []
        for idx in range(len(i)):
            tmp = [0, 0]
            tmp[0] = i[idx]
            tmp[1] = j[idx]
            indices.append(tmp)
        for layer in range(self.BP_layers):
            values = tf.Variable(np.ones(len(i)), dtype=tf.float32, name=format("v_to_c_sparse_var_%d" % layer))
            self.V_to_C_params[layer] = tf.SparseTensor(indices=indices, values=values
                                                        , dense_shape=self.H_sumV_to_C.shape)

        # --------------------不带训练参数的BP译码网络------------------
        if 1 == placement or ((not self.train_bp_network) and (not self.use_train_bp_net)):
            self.llr_into_bp_net, self.xe_0, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out, self.sigmoid_out = self.build_network()
            self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            print("open a tf session")
            self.sess.run(init)
            return
        # -----------------带训练参数的BP译码网络的参数矩阵--------------
        else:  # 之后考虑为每个 H_sumC_to_V 和 H_sumV_to_C 单独进行变量随机化
            # ---------改为 tf.Variable ----------
            self.H_x_to_xe0 = tf.Variable(self.H_x_to_xe0, dtype=tf.float32, name="H_x_to_xe0")
            # self.H_sumV_to_C = tf.Variable(self.H_sumV_to_C, dtype=tf.float32)
            self.H_xe_v_sumc_to_y = tf.Variable(self.H_xe_v_sumc_to_y, dtype=tf.float32, name="H_xe_v_sumc_to_y")
            self.llr_into_bp_net, self.xe_0, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out, self.logits, self.bp_out_llr = self.build_trained_bp_network()
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder
        # self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_into_bp_net))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder.

        # self.cross_entropy = -tf.reduce_sum(self.llr_into_bp_net * tf.log(self.sigmoid_out), 1)  # * 是按元素相乘，u_coded_bits=(5000,6);sigmoid_out=(6,5000)
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits, name="cross_entroy")  # * 是按元素相乘，u_coded_bits=(5000,6);sigmoid_out=(6,5000)
        self.cross_entropy = tf.reduce_sum(self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.cross_entropy)

        self.sess = tf.Session(graph=tf.get_default_graph())  # open a session
        # ---- tmp print --------------
        # print(self.sess.run((self.C_to_V_params[0])))
        print('Open a tf session!')
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 恢复已经训练过的 bp_net 参数
        if placement == 0:
            self.bp_net_save_dir = format("model/bp_model/%s_%s/BP%s/before_cnn/" % (top_config.N_code, top_config.K_code, self.BP_layers))
        else:
            self.bp_net_save_dir = format("model/bp_model/%s_%s/BP%s/after_cnn/" % (top_config.N_code, top_config.K_code, self.BP_layers))
        self.bp_model = format("bp_model_BP%s.ckpt_10000" % self.BP_layers)
        # 如果已经训练过了，并且需要训练，就先加载之前的参数
        if self.use_train_bp_net and os.path.isfile(self.bp_net_save_dir + self.bp_model + ".meta"):
            saver = tf.train.Saver()
            saver.restore(self.sess, self.bp_net_save_dir + self.bp_model)

    def __del__(self):
        if self.train_bp_network or self.use_train_bp_net:
            self.sess.close()
        print('Close a tf session!')


    def atanh(self, x):
        x1 = tf.add(1.0, x)
        x2 = tf.subtract((1.0), x)
        x3 = tf.divide(x1, x2)
        x4 = tf.log(x3)
        return tf.divide(x4, (2.0))

    def one_bp_iteration(self, xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, xe_0):
        """
        :param xe_v2c_pre_iter: (2040, 5000) ,xe_v2c_pre_iter 是上一轮的变量节点，即非初始化的变量节点
        :param H_sumC_to_V: (2040, 2040) 纵向排列调整为横向排列，同时横向更新
        :param H_sumV_to_C: (2040, 2040) 横向排列调整为纵向排列，同时纵向更新
        :param xe_0: (2040, 5000) 初始化的变量节点
        :return: 
        """
        for layer in range(self.BP_layers):
            if 0 != layer:
                xe_v2c_pre_iter = xe_c_sumv

            xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
            xe_tanh = tf.to_float(xe_tanh)
            xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
            xe_sum_log_img = tf.matmul(H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
            xe_sum_log_real = tf.matmul(H_sumC_to_V, tf.log(1e-8 + tf.abs(xe_tanh)))
            xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
            xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
            xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
            xe_pd_modified = tf.add(xe_product, xe_product_temp)
            xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
            xe_c_sumv = tf.add(xe_0, tf.matmul(H_sumV_to_C, xe_v_sumc))
        return xe_v_sumc, xe_c_sumv  # xe_v_sumc 是输出层，xe_c_sumv 是这一轮BP的输出，下一轮的输入

    def H(self, x):
        ex = tf.exp(x)
        return tf.log(tf.truediv(1 + ex, 1 - ex))

    def multiple_bp_iteration(self, xe_v2c_pre_iter, xe_0):
        """
        :param xe_v2c_pre_iter: (2040, 5000) ,xe_v2c_pre_iter 是上一轮的变量节点，即非初始化的变量节点
        :param H_sumC_to_V: (2040, 2040) 纵向排列调整为横向排列，同时横向更新
        :param H_sumV_to_C: (2040, 2040) 横向排列调整为纵向排列，同时纵向更新
        :param xe_0: (2040, 5000) 初始化的变量节点
        :return: 
        """
        # ------------- 创建可训练权重矩阵 -----------------------
        for layer in range(self.BP_layers):
            if 0 != layer:
                xe_v2c_pre_iter = xe_c_sumv

            # xe_vml = tf.exp(xe_v2c_pre_iter)
            # xe_vml_ln = tf.log(tf.truediv(tf.add(xe_vml, - 1), tf.add(xe_vml, 1)))
            # xe_vml_ln_sum = tf.sparse_tensor_dense_matmul(self.C_to_V_params[layer], xe_vml_ln)
            # xe_v_sumc = self.H(xe_vml_ln_sum)
            # xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.V_to_C_params[layer], xe_v_sumc))

        # ----------------------------
            xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0], name=format("truediv_xe_pre_2_%d" % layer)), name=format("to_double_%d" % layer)), name=format("xe_tanh1_%d" % layer))  # 除法 tanh(ve_v3c_pre_iter/2.0)
            xe_tanh = tf.to_float(xe_tanh, name=format("xe_tanh2_%d" % layer))
            xe_tanh_temp = tf.sign(xe_tanh, name=format("xe_tanh_temp_%d" % layer))  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
            xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.C_to_V_params_img[layer], tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926], name=format("multiply_in_sparse_%d" % layer)), name=format("xe_sum_log_img_%d" % layer))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
            xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.C_to_V_params[layer], tf.log(1e-8 + tf.abs(xe_tanh), name=format("log_in_sparse_%d" % layer)), name=format("xe_sum_log_real_%d" % layer))
            xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img, name=format("xe_sum_log_complex_%d" % layer))
            xe_product = tf.real(tf.exp(xe_sum_log_complex), name=format("xe_product_%d" % layer))  # xe_sum_log_real
            xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7, name=format("xe_product_temp_%d" % layer))
            xe_pd_modified = tf.add(xe_product, xe_product_temp, name=format("xe_pd_modified_%d" % layer))
            xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0], name=format("xe_v_sumc_%d" % layer))
            xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.V_to_C_params[layer], xe_v_sumc), name=format("xe_c_sumv_%d" % layer))

        return xe_v_sumc, xe_c_sumv  # xe_v_sumc 是输出层，xe_c_sumv 是这一轮BP的输出，下一轮的输入

    def build_network(self):  # build the network for one BP iteration
        # 还需要构建一段由 u_coded_bits 和 SNR 到 llr 的网络。
        # BP initialization
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32))  # 建立了一个矩阵变量（576 * 5000)，576 是码元，5000是每次5000个码元为一个batch
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net)  # 横向edge初始化(H_x_to_xe0:shape=(2040, 576), llr_into_bp_net:shape=(576, 5000) => (2040, 5000)
        xe_v2c_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32))  # the v->c messages of the previous iteration, shape=(2040, 5000)
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe_0)  # 将 xe_0 赋值给 ve_v2c_pre_iter_assign

        # one iteration
        H_sumC_to_V = tf.constant(self.H_sumC_to_V, dtype=tf.float32)  # shape=(2040, 2040)
        H_sumV_to_C = tf.constant(self.H_sumV_to_C, dtype=tf.float32)  # shape=(2040, 2040)
        # 上面两个变量改成下面的这两句
        # H_sumC_to_V = self.H_sumC_to_V
        # H_sumV_to_C = self.H_sumV_to_C
        # --------------------------
        xe_v_sumc, xe_c_sumv = self.one_bp_iteration(xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, xe_0)  # (2040, 5000), (2040, 2040), (2040, 2040), (2040, 5000)
        # xe_v_sumc 是纵向排列的edge，xe_c_sumv 是横向排列的edge
        # 横向排列的edge正好是每轮BP的输出，而纵向排列的BP则是可以作为输出层的前一个数据层
        # start the next iteration
        start_next_iteration = xe_v2c_pre_iter.assign(xe_c_sumv)

        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))  # H_xe_sumc_to_y 是输出层的转换矩阵，xe_v_sumc 是纵向排列的edge
        # sigmoid_out = tf.sigmoid(tf.transpose(bp_out_llr))
        sigmoid_out = tf.sigmoid(bp_out_llr)
        dec_out = tf.transpose(tf.floordiv(1-tf.to_int32(tf.sign(bp_out_llr)), 2), name="output_node_tanspose")

        return llr_into_bp_net, xe_0, xe_v2c_pre_iter_assign, start_next_iteration, dec_out, sigmoid_out

    def build_trained_bp_network(self):  # build the network for one BP iteration
        # 还需要构建一段由 u_coded_bits 和 SNR 到 llr 的网络。
        # BP initialization
        # llr_into_bp_net = tf.placeholder(dtype=tf.float32, shape=[self.v_node_num, self.batch_size], name="llr_into_bp_net_tensor")  # 整个BP网络的输入
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32), name="llr_into_bp_net")  # 建立了一个矩阵变量（576 * 5000)，576 是码元，5000是每次5000个码元为一个batch
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net, name="xe_0")  # 横向edge初始化(H_x_to_xe0:shape=(2040, 576), llr_into_bp_net:shape=(576, 5000) => (2040, 5000)
        xe_v2c_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32), name="xe_v2c_pre_iter")  # the v->c messages of the previous iteration, shape=(2040, 5000)
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe_0, name="xe_v2c_pre_iter_assign")  # 将 xe_0 赋值给 ve_v2c_pre_iter_assign

        xe_v_sumc, xe_c_sumv = self.multiple_bp_iteration(xe_v2c_pre_iter, xe_0)  # (2040, 5000), (2040, 2040), (2040, 2040), (2040, 5000)
        self.xe_v_sumc = xe_v_sumc
        # xe_v_sumc 是纵向排列的edge，xe_c_sumv 是横向排列的edge
        # 横向排列的edge正好是每轮BP的输出，而纵向排列的BP则是可以作为输出层的前一个数据层
        # start the next iteration
        # start_next_iteration = xe_v2c_pre_iter.assign(xe_c_sumv)
        start_next_iteration = 0
        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))  # H_xe_sumc_to_y 是输出层的转换矩阵，不需要训练，xe_v_sumc 是纵向排列的edge
        # sigmoid_out = tf.sigmoid(tf.transpose(bp_out_llr))
        logits = tf.transpose(bp_out_llr)  # logits 替换上面的 sigmoid_out
        # sigmoid_out = tf.sigmoid(bp_out_llr)
        dec_out = tf.transpose(tf.floordiv(1-tf.to_int32(tf.sign(bp_out_llr)), 2), name="output_node_tanspose")

        return llr_into_bp_net, xe_0, xe_v2c_pre_iter_assign, start_next_iteration, dec_out, logits, bp_out_llr

    '''
    python -m tensorflow.python.tools.freeze_graph --input_checkpoint=model/bp_model/bp_model.ckpt --input_binary=false --output_graph=model/bp_model/frozen.pb --input_graph=model/bp_model/bp_model.pbtxt --output_node_names=output_node_tanspose
    '''
    # 可训练的BP decode
    def decode(self, llr_in, bp_iter_num):
        real_batch_size, num_v_node = np.shape(llr_in)  # llr_in 就是BP译码的初始化
        if real_batch_size != self.batch_size:  # padding zeros
            llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
            # original llr array.
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})  # llr应该只是数据层

        # 每个作用域内 tensor 都需要初始化一下
        # init = tf.global_variables_initializer()
        # self.sess.run(init)

        # 尝试保存网络
        # saver = tf.train.Saver()
        # save_dir = "model/bp_model/"

        self.sess.run(self.xe_v2c_pre_iter_assign)  # BP 网络的第一层
        # for iter in range(0, bp_iter_num-1):
        #     self.sess.run(self.start_next_iteration)  # run start_next_iteration时表示当前一轮BP的输出
        y_dec = self.sess.run(self.dec_out)  # dec_out 则是最终输出层
        # sigmoid_out = self.sess.run(self.sigmoid_out)
        # saver.save(self.sess, save_dir + "bp_model.ckpt")

        if real_batch_size != self.batch_size:
            y_dec = y_dec[0:real_batch_size, :]

        return y_dec

    # # 不可训练的BPdecode
    # def decode(self, llr_in, bp_iter_num):
    #     real_batch_size, num_v_node = np.shape(llr_in)
    #     if real_batch_size != self.batch_size:  # padding zeros
    #         llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
    #         # original llr array.
    #     self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
    #     self.sess.run(self.xe_v2c_pre_iter_assign)
    #     for iter in range(0, bp_iter_num-1):
    #         self.sess.run(self.start_next_iteration)
    #     y_dec = self.sess.run(self.dec_out)
    #     if real_batch_size != self.batch_size:
    #         y_dec = y_dec[0:real_batch_size, :]
    #
    #     return y_dec

    def train_decode_network(self, bp_iter_num, SNRset, batch_size, ch_noise_normalize, linear_code):
        """
        神经网络的：
        输入 u_coded_bits 和 SNR
        输出 LLR
        :param llr_in: 这里 LLR 是和 u_coded_bits 相反的 
        :param bp_iter_num: 
        :param u_coded_bits: 处理方法是将 u_coded_bits 中的 0 和 1 翻过来 
        :return: 
        """
        noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file)

        # 尝试保存网络
        saver = tf.train.Saver(max_to_keep=20)
        G_matrix = linear_code.G_matrix  # 用于产生 u_coded_bits 样本的生成矩阵
        LLR = []
        z = 0
        u_coded_bits = []
        for i in range(600):  # 每一种SNR的训练轮数，原来是 20000
            for SNR in SNRset:
                real_batch_size = batch_size
                # 需要一个更新输入数据的过程
                x_bits, u_coded_bits, s_mod, channel_noise, y_receive, LLR, ch_noise_sigma = lbc.encode_and_transmission(G_matrix, SNR, real_batch_size, noise_io)
                # --------------------------------------------------------------------------------------------
                # x = self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: LLR})
                # y = self.sess.run(self.llr_into_bp_net)
                # z = self.llr_into_bp_net.eval(self.sess)
                # v = self.sess.run(self.sigmoid_out)
                # x2 = self.sess.run(self.bp_out_llr)
                # x3 = self.sess.run(self.xe_v_sumc)
                # self.sess.run()  # 重新修改网络的输入为 llr_in
                # p = self.sess.run(self.cross_entropy, feed_dict={self.labels: u_coded_bits})
                self.sess.run([self.llr_assign, self.train_step], feed_dict={self.llr_placeholder: LLR, self.labels: u_coded_bits})

                # y = self.sess.run(self.llr_into_bp_net)
                # x,y = self.sess.run([])
                # -tf.reduce_sum(self.llr_into_bp_net * tf.log(self.sigmoid_out))
                # x1 = self.sess.run(tf.log(self.sigmoid_out))
                pass
            # if 0 == (i % 100):
            #     print(z)
            # if 0 == i % 5000 and 0 != i:
            #     saver.save(self.sess, self.bp_net_save_dir + self.bp_model + format("_%d" % i))
            #     print("this num %d epo" % i)

        # ------------- 保存训练好的神经网络 -----------------
        saver.save(self.sess, self.bp_net_save_dir + self.bp_model)
        print(self.sess.run(tf.sparse.to_dense(self.V_to_C_params[0])))

    def generate_inputs(self, G_matrix, batch_size, rng):
        K, N = np.shape(G_matrix)
        if rng == 0:
            x_bits = np.random.randint(0, 2, size=(batch_size, K))
        else:
            x_bits = rng.randint(0, 2, size=(batch_size, K))  # 随机数种子rng用于生产随机的输入码字
        # coding
        u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # G_matrix
        return u_coded_bits
 # [[{{node beta2_power/read}}]]