import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False

def sigmoid(x):
    result = 1 / (1 + math.e ** (-x))
    return result

def tanh(x):
    # result = np.exp(x)-np.exp(-x)/np.exp(x)+np.exp(-x)
    result = (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))
    return result

def relu(x):
    result = np.maximum(0, x)
    return result

fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
x = np.linspace(-10, 10, 100)
y = relu(x)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([5, 10])
ax.set_yticks([-1, 1])
ax.set_ylim([-1.5, 1.5])

plt.plot(x, y, label="Relu", linestyle='-', color='black')
plt.legend()

# ax = fig.add_subplot(122)
x = np.linspace(-10, 10)
y = tanh(x)

# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
#
# ax.set_xticks([-10, -5, 0, 5, 10])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([-1, -0.5, 0.5, 1])

plt.plot(x, y, label="Tanh", linestyle='--', color='black')
plt.legend()
# plt.savefig('sigmoid and tanh.png', dpi=200)

# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(121)
x = np.linspace(-10, 10)
y = sigmoid(x)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.set_yticks([-1, -0.5, 0.5, 1])

plt.plot(x, y, label="Sigmoid", linestyle='-.', color='black')
plt.legend()

plt.show()
