# import tensorflow as tf
# with tf.Session() as sess:
#        # 导入 .chpt.meta 文件，其实就是将 ckpt 加载到默认图中
#     _ = tf.train.import_meta_graph("../../Iterative-BP-CNN/model/netid0_model0/model.ckpt.meta", clear_devices=True)
#        # 没有定义图，意味着上面的操作都是在默认图上的，因此这边拿出默认图
#     g = tf.get_default_graph()
#     # 将默认图g，输出为pb/pbtxt文件。
#        # 参数：g是默认图，"."输出到当前目录，False表示输出二进制（即.pb)，True表示文本（即.pbtxt)
#     tf.train.write_graph(g,".", "graph.pb", False)
import tensorflow as tf

ckpt_path = "./bp_model/576_432/bp_model.ckpt_10000.meta"
out_graph_name = "./bp_model/576_432/bp_model_10000.pbtxt"

with tf.Session() as sess:
    _ = tf.train.import_meta_graph(ckpt_path, clear_devices=True)
    g = tf.get_default_graph()
    tf.train.write_graph(g, ".", out_graph_name, True)