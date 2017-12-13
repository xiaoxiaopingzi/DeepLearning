# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: 2.11-Vectorization.py 
@time: 2017-11-09 21:24  
"""
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print("Vectorized version:" + str(1000*(toc - tic)) + "ms")

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("Non-Vectorized version(For loop):" + str(1000*(toc - tic)) + "ms")
