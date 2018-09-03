# l = [1, 2, 3]
# l.pop(2)
# print(l)

# a = 3
# a <<= 0
# print(a)

# l = [1, 2, 3, 4, 5]
# for i in range(3, 0, -1):
#     print(l[i])

import tensorflow as tf
# a = [[[1, 1, 1, 1], [1, 1, 1, 1],
#      [2, 2, 2, 2], [2, 2, 2, 2],
#      [3, 3, 3, 3], [3, 3, 3, 3],]]
# b = [[[1, 1, 1, 1], [1, 1, 1, 1],
#      [2, 2, 2, 2], [2, 2, 2, 2],
#      [3, 3, 3, 3], [3, 3, 3, 3],]]
# c = tf.concat([a, b], axis=-1)
# print(c)
# target = [[0, 1, 2, 0,], [0, 1, 2, 0]]
# batch_size = 2
# nums_tags = 8
# ones = tf.cast(nums_tags * tf.ones([batch_size, 1]), tf.int32)
# print(ones)
# two = tf.concat([ones, target], axis=-1)
# print(two)

# CRF层中计算scores的源码
# 状态得分
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
# input.shape=[batch_size, max_seq_len, num_tags]=[3,4,3]
sess = tf.Session()
inputs = [[[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # 预测标签
          [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
# 通过字典和numpy操作加入权重参数
# inputs_tensor = tf.convert_to_tensor(inputs)
# inputs_numpy = inputs_tensor.eval(session=sess)
# print("offsets_numpy:", inputs_numpy)
# weight_dict = {0:1, 1:1, 2:2, 3:1}
# for i in range(len(inputs_numpy)):
#     for j in range(len(inputs_numpy[i])):
#         inputs_numpy[i][j] *= weight_dict[np.argmax(inputs_numpy[i][j])]
# print(inputs_numpy)
# inputs = tf.convert_to_tensor(inputs_numpy)
# 通过tensor点乘加入权重参数
# inputs = tf.convert_to_tensor(inputs)
# weight_matrix = [[[1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1]],
#                  [[1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1]],
#                  [[1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 2, 1],[1, 1, 2, 1]]]
# weight_matrix = tf.convert_to_tensor(weight_matrix)
# inputs = inputs*weight_matrix
# print("inputs", inputs)
# 通过字典初始化权重参数矩阵，然后通过tensor点乘加入权重参数
# inputs = tf.convert_to_tensor(inputs)
# weight_list = [1, 1, 2, 1]
# weight_matrix_numpy = np.ones([3, 4, 4])
# for i in range(len(weight_matrix_numpy)):
#     for j in range(len(weight_matrix_numpy[i])):
#         weight_matrix_numpy[i][j] = weight_list
# print("weight_matrix_numpy:", weight_matrix_numpy)
# weight_matrix = tf.convert_to_tensor(weight_matrix_numpy)
# weight_matrix = tf.cast(weight_matrix, "int32")
# inputs = inputs*weight_matrix
# 用tensor完成权重参数的添加
inputs = tf.convert_to_tensor(inputs)
batch_size = array_ops.shape(inputs)[0]
max_seq_len = array_ops.shape(inputs)[1]
num_tags = array_ops.shape(inputs)[2]
weight_matrix = tf.tile([[[1, 1, 1, 1]]], [batch_size, max_seq_len, 1])
weight_matrix = tf.reshape(weight_matrix, [batch_size, max_seq_len, num_tags])
weight_matrix_numpy = weight_matrix.eval(session=sess)
# print("weight_matrix_numpy:", weight_matrix_numpy)
inputs = inputs*weight_matrix
# tag_indices=[batch_size, max_seq_len]=[3,4]
tag_indices = [[3, 2, 2, 3],    # 真实标签
               [3, 2, 2, 3],
               [3, 2, 2, 3]]
batch_size = array_ops.shape(inputs)[0]
max_seq_len = array_ops.shape(inputs)[1]
num_tags = array_ops.shape(inputs)[2]
# print(batch_size, max_seq_len, num_tags)
flattened_inputs = array_ops.reshape(inputs, [-1])
flattened_inputs_numpy = flattened_inputs.eval(session=sess)    # 将tensor转换成array的方法
print("flattened_inputs_numpy:", flattened_inputs_numpy)
offsets = array_ops.expand_dims(math_ops.range(batch_size) * max_seq_len * num_tags, 1)
offsets_numpy = offsets.eval(session=sess)
print("offsets_numpy:", offsets_numpy, type(offsets_numpy))
offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
offsets_numpy2 = offsets.eval(session=sess)
print("offsets_numpy2:", offsets_numpy2)
flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])
flattened_tag_indices_numpy = flattened_tag_indices.eval(session=sess)
print("flattened_tag_indices_numpy:", flattened_tag_indices_numpy)
# 开始计算scores
# 根据flattened_tag_indices的值寻找flattened_inputs中相应索引的值的和，非常巧妙的方法，
# 因为flattened_tag_indices中的值就是正确标签所在的位置
unary_scores1 = array_ops.gather(flattened_inputs, flattened_tag_indices)
unary_scores1_numpy = unary_scores1.eval(session=sess)
print("unary_scores1_numpy:", unary_scores1_numpy)
unary_scores2 = array_ops.reshape(unary_scores1, [batch_size, max_seq_len])
unary_scores2_numpy = unary_scores2.eval(session=sess)
print("unary_scores2_numpy:", unary_scores2_numpy)
unary_scores_end = math_ops.reduce_sum(unary_scores2, 1)
unary_scores_end_numpy = unary_scores_end.eval(session=sess)
print("unary_scores_end_numpy:", unary_scores_end_numpy)
print("===========================================================================")
# 状态转移得分(不需要预测的标签，只需要用真实标签来构建状态转移概率矩阵)
transition_params = [[0, 1, 1, 2], [1, 2, 0, 1], [0, 1, 1, 2], [1, 2, 0, 1]]    # 状态转移概率矩阵初始化
tag_indices = [[3, 2, 2, 3],    # 真实标签
               [3, 2, 2, 3],
               [3, 2, 2, 3]]
transition_params = tf.convert_to_tensor(transition_params)
num_tags = transition_params.get_shape()[0]
num_transitions = array_ops.shape(tag_indices)[1] - 1   # transition_params列数-1
print("num_transitions:", num_transitions.eval(session=sess))
start_tag_indices = array_ops.slice(tag_indices, [0, 0], [-1, num_transitions])  # transition_params中除去最后一列
print("start_tag_indices:", start_tag_indices.eval(session=sess))
end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])   # transition_params中除去第一列
print("end_tag_indices:", end_tag_indices.eval(session=sess))
flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices   #
print("flattened_transition_indices:", flattened_transition_indices.eval(session=sess))
flattened_transition_params = array_ops.reshape(transition_params, [-1])
print("flattened_transition_params:", flattened_transition_params.eval(session=sess))
# 根据状态转移概率矩阵和真实标签之间的转移关系进行打分，依然是用flattened_transition_indices中的值去flattened_transition_params中
# 找对应索引的值，同样非常巧妙，因为flattened_transition_indices中的值代表一种转移关系，
# flattened_transition_params在该索引处的值就是就是相应状态转移概率(分数)，
# 若真实标签中经常出现的状态转移关系在状态转移概率矩阵中概率越高、很少出现甚至从未出现的转移关系在状态转移概率矩阵中概率越低，
# 则预测得分就会高，所以经过训练后，
binary_scores = array_ops.gather(flattened_transition_params, flattened_transition_indices)
print("binary_scores:", binary_scores.eval(session=sess))
