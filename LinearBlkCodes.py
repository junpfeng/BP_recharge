import numpy as np
import tensorflow as tf

# Generate some random bits, encode it to valid codewords and simulate transmission
def encode_and_transmission(G_matrix, SNR, batch_size, noise_io, rng=0):
    K, N = np.shape(G_matrix)
    if rng == 0:
        x_bits = np.random.randint(0, 2, size=(batch_size, K))
    else:
        x_bits = rng.randint(0, 2, size=(batch_size, K))  # 随机数种子rng用于生产随机的输入码字
    # coding
    u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # G_matrix

    # BPSK modulation
    s_mod = u_coded_bits * (-2) + 1  # 对码元做了颠倒了，0->1,1->-1，去除了直流分量
    # plus the noise
    ch_noise_normalize = noise_io.generate_noise(batch_size)  # 生成均值为0，方差为1的高斯随机噪声矩阵（相干性设为0）

    #ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)  # SNR = 10*lg(1/(2*sigma^2))
    ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0))  # SNR = 10*lg(1/(2*sigma^2))
    ch_noise = ch_noise_normalize * ch_noise_sigma  # 利用公式 D(c*x) = c^2*D(x) 来改变噪声的方差
    y_receive = s_mod + ch_noise  # 模拟接收信号
    LLR = y_receive * 2.0 / (ch_noise_sigma * ch_noise_sigma)  # 计算接收信号的对数似然比
    # ---------------------------------------------------------------------------
    return x_bits, 1 - u_coded_bits, s_mod, ch_noise, y_receive, LLR, ch_noise_sigma  #, SNR, u_coded_bits_tensor, LLR_tensor, SNR_tensor, ch_noise_normalize
    # s_mod 是经 BPSK 调制后的信号(0，1)-> (-1,1)，在调制完成后，还做了一个翻转
    # x_bits 随机生成的发送端码元，u_coded_bits 对x_bits做纠错编码后的码元，s_mod 对u_coded_bits做BPSK调制后的码元，ch_noise 信道噪声，y_recive 接收端接收到的信号，LLR 对数似然比
    # u_coded_bits 是经过构造函数之后的码字，s_mod 是经过BPSK调制的码字，LLR是对数似然比输入，第 2 个、第 3 个和第 6 个。
class LDPC:
    def __init__(self, N, K, file_G, file_H):
        self.N = N
        self.K = K
        self.G_matrix, self.H_matrix = self.init_LDPC_G_H(file_G, file_H)

    def init_LDPC_G_H(self, file_G, file_H):
        G_matrix_row_col = np.loadtxt(file_G, dtype=np.int32)
        H_matrix_row_col = np.loadtxt(file_H, dtype=np.int32)
        G_matrix = np.zeros([self.K, self.N], dtype=np.int32)
        H_matrix = np.zeros([self.N - self.K, self.N], dtype=np.int32)
        G_matrix[G_matrix_row_col[:, 0], G_matrix_row_col[:, 1]] = 1
        H_matrix[H_matrix_row_col[:, 0], H_matrix_row_col[:, 1]] = 1
        return G_matrix, H_matrix

    def dec_src_bits(self, bp_output):
        return bp_output[:,0:self.K]

