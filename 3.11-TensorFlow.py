# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: 3.11-TensorFlow.py 
@time: 2017-11-19 14:12  
"""
import numpy as np
import tensorflow as tf
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

coefficients = np.array([[1], [-20], [25]])

w = tf.Variable([0], dtype=tf.float32)
x = tf.placeholder(tf.float32, [3, 1])

cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]  # (w-5)**2
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
# print(session.run(w))

start_time = time.time()
for i in range(1000):
    session.run(train, feed_dict={x: coefficients})
end_time = time.time()

print(session.run(w))
print("用时%d毫秒" % ((end_time - start_time) * 1000))