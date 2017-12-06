# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: HelloTensorFlow.py 
@time: 2017-11-19 11:19  
"""
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant("hello, TensorFlow")
session = tf.Session()
print(session.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print(session.run(a+b))

