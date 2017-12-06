# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: numpy_1.py 
@time: 2017-12-06 10:32  
"""
import numpy as np

# 读取txt文件中的数据，生成数组
world_alcohol = np.genfromtxt("world_alcohol.txt", delimiter=",", dtype=str)
print(type(world_alcohol))  # 类型为：<class 'numpy.ndarray'>
print(world_alcohol)
# 打印矩阵中的元素，注意元素的索引是从0开始的(行的索引以及列的索引都是从0开始的)
print("第0行第0个元素是:" + world_alcohol[0, 5])
# print(help(np.genfromtxt))  # 打印出函数的文档

# numpy中list中的元素必须是同一个类型的，如果输入的元素不一致，numpy会自动转换
# vector = np.array([5, 10, 15, 20.0])  # 生成行向量
vector = np.array([5, 10, 15, 20])  # 生成行向量
matrix = np.array([[5, 10, 15], [20, 25, 30]])  # 生成数组

print(vector)
print(vector[0:3])  # python中的切片，包括头不包括尾
print(type(vector))  # 输出vector的类型
print(vector.dtype)  # 输出vector中元素的类型
print(vector.shape)  # 输出行向量中元素的个数

print(matrix)
print(type(matrix))
print(matrix.shape)  # 输出矩阵的维数，即2 x 3
print(matrix[:, 1])  # 用冒号占位符表示取所有元素，这里表示取矩阵第1列的所有元素
print(matrix[:, 0:2])

print(vector == 10.0)  # 判断向量中的元素是否和10相等
print(matrix == 10)

equal_to_ten = (vector == 10.0)
print(equal_to_ten)
print(vector[equal_to_ten])  # numpy会过滤掉为false的索引，只输出索引为true的索引

equal = (vector == 10.0) | (vector == 5.0)
print(equal)
print(vector[equal])

vector2 = np.array(["1", "2", "3"])
print(vector2)
print(vector2.dtype)
vector2 = vector2.astype(float)  # numpy中的类型转换，将字符型数据转换为float型数据
print(vector2)
print(vector2.dtype)

# 取矩阵或者向量的最大值或者最小值
print(vector.min())
print("向量的最大值为：%d" % vector.max())
print("矩阵的最大值为：%d" % matrix.max())

print("对矩阵按行求和：")
print(matrix)
print(matrix.sum(axis=1))
print(matrix.sum(axis=0))  # 对矩阵按列求和
