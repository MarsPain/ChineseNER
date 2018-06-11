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

import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
#input.shape=[batch_size, max_seq_len, num_tags]=[3,4,3]
sess = tf.Session()
inputs = [[[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
inputs_tensor= tf.convert_to_tensor(inputs)
inputs_numpy = inputs_tensor.eval(session=sess)
print("offsets_numpy:", inputs_numpy)
weight_dict = {0:1, 1:1, 2:2, 3:1}
for i in range(len(inputs_numpy)):
    for j in range(len(inputs_numpy[i])):
        inputs_numpy[i][j] *= weight_dict[np.argmax(inputs_numpy[i][j])]
print(inputs_numpy)
inputs= tf.convert_to_tensor(inputs_numpy)
#tag_indices=[batch_size, max_seq_len]=[3,4]
tag_indices = [[3, 2, 2, 3],
               [3, 2, 2, 3],
               [3, 2, 2, 3]]
batch_size = array_ops.shape(inputs)[0]
max_seq_len = array_ops.shape(inputs)[1]
num_tags = array_ops.shape(inputs)[2]
# print(batch_size, max_seq_len, num_tags)
flattened_inputs = array_ops.reshape(inputs, [-1])
flattened_inputs_numpy = flattened_inputs.eval(session=sess)
print("flattened_inputs_numpy:", flattened_inputs_numpy)
offsets = array_ops.expand_dims(math_ops.range(batch_size) * max_seq_len * num_tags, 1)
offsets_numpy = offsets.eval(session=sess)  #将tensor转换成array的方法
print("offsets_numpy:", offsets_numpy, type(offsets_numpy))
offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
offsets_numpy2 = offsets.eval(session=sess)
print("offsets_numpy2:", offsets_numpy2)
flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])
flattened_tag_indices_numpy = flattened_tag_indices.eval(session=sess)
print("flattened_tag_indices_numpy:", flattened_tag_indices_numpy)
#开始计算scores
#根据flattened_tag_indices的值寻找flattened_inputs中相应索引的值，非常巧妙的方法
unary_scores1 = array_ops.gather(flattened_inputs, flattened_tag_indices)
unary_scores1_numpy = unary_scores1.eval(session=sess)
print("unary_scores1_numpy:", unary_scores1_numpy)
unary_scores2 = array_ops.reshape(unary_scores1, [batch_size, max_seq_len])
unary_scores2_numpy = unary_scores2.eval(session=sess)
print("unary_scores2_numpy:", unary_scores2_numpy)
unary_scores_end = math_ops.reduce_sum(unary_scores2, 1)
unary_scores_end_numpy = unary_scores_end.eval(session=sess)
print("unary_scores_end_numpy:", unary_scores_end_numpy)
