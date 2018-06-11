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
target = [[0, 1, 2, 0,], [0, 1, 2, 0]]
batch_size = 2
nums_tags = 8
ones = tf.cast(nums_tags * tf.ones([batch_size, 1]), tf.int32)
print(ones)
two = tf.concat([ones, target], axis=-1)
print(two)