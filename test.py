# -*- coding: utf-8 -*-)
import tensorflow as tf

# 在系统默认计算图上创建张量和操作
a = tf.constant([1.0, 2.0])
b = tf.constant([2.0, 1.0])
result = a + b

# 定义两个计算图
g1 = tf.Graph()
g2 = tf.Graph()

# 在计算图g1中定义张量和操作
with g1.as_default():
    a = tf.constant([1.0, 1.0])
    b = tf.constant([1.0, 1.0])
    result1 = a + b

with g2.as_default():
    a = tf.constant([2.0, 2.0])
    b = tf.constant([2.0, 2.0])
    result2 = a + b

# 在g1计算图上创建会话
# with tf.Session(graph=g1) as sess:
#     out = sess.run(result1)
#     print('with graph g1, result: {0}'.format(out))
init = tf.global_variables_initializer()
sess = tf.Session(graph=tf.get_default_graph())
sess.run(init)
out = sess.run(result)
print('with graph g, result: {0}'.format(out))

sess1 = tf.Session(graph=g1)
out = sess1.run(result1)
print('with graph g1, result: {0}'.format(out))

sess2 = tf.Session(graph=g2)
out = sess2.run(result2)
print('with graph g2, result: {0}'.format(out))



# with tf.Session(graph=g2) as sess:
#     out = sess.run(result2)
#     print ('with graph g2, result: {0}'.format(out))
#
# # 在默认计算图上创建会话
# with tf.Session(graph=tf.get_default_graph()) as sess:
#     out = sess.run(result)
#     print('with graph default, result: {0}'.format(out))

print (g1.version)  # 返回计算图中操作的个数
