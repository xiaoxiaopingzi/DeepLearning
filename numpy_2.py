# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: numpy_2.py 
@time: 2017-12-08 15:32  
"""
# 注意在numpy中，如果参数是2个，则这个两个参数很大可能需要组成一个元组
import numpy as np

B = np.arange(3)
print(B)
print(np.exp(B))  # 输出矩阵B的e次方
print(np.sqrt(B))  # 输出矩阵B的平方根

# floor函数表示向下取整
a = np.floor(10 * np.random.random((3, 4)))
print(a)
print("---------将矩阵展开成一个向量---------")
print(a.ravel())  # 将矩阵展开成一个向量
print("------------将向量转换成矩阵------------")
a.shape = (6, 2)  # 将向量重新转换为6 x 2的矩阵
print(a)
print("-----------输出矩阵的转置----------")
print(a.T)  # a.T表示矩阵a的转置

# 对矩阵进行合并
a1 = np.floor(10 * np.random.random((2, 2)))
a2 = np.floor(10 * np.random.random((2, 2)))
print("-------矩阵a1为------")
print(a1)
print("---------矩阵a2为----------")
print(a2)
print("-------将矩阵a1和a2按行拼接，得到的新矩阵为--------------")
print(np.vstack((a1, a2)))  # 将矩阵a1和a2按行拼接，a1在前，a2在后

print("-------将矩阵a1和a2按列拼接，得到的新矩阵为--------------")
print(np.hstack((a1, a2)))  # 将矩阵a1和a2按列拼接，a1在前，a2在后

# 对矩阵进行切分
a3 = np.floor(10 * np.random.random((2, 12)))
print("------矩阵a3为--------")
print(a3)
print("----------将矩阵按照列平均分为3份-------------")
print(np.hsplit(a3, 3))
print("----------将矩阵按照索引进行切分，在矩阵第3列和第4列的前面都切一刀----------------------")
print(np.hsplit(a3, (3, 4)))

a4 = np.floor(10 * np.random.random((12, 2)))
print("----------将矩阵按照行平均分为3份---------------")
print(np.vsplit(a4, 3))

# 在下面的情况下，a5和a6是完全一样的，更改a6会导致a5也发生改变
print("-------使用a=b的复制方式会导致a和b是完全相同的对象------")
a5 = np.arange(12)
a6 = a5
print(a6 is a5)
a6.shape = (3, 4)
print(a5.shape)
print(id(a5))
print(id(a6))  # a5和a6是完全一样的

print("-------使用a=b.copy()的复制方式会导致a和b是完全不同的对象------")
# 在下面的情况下，a7和a8是完全不同的对象
a7 = np.arange(12)
a8 = a7.copy()
print(a8 is a7)
a8[8] = 1234
print(a7)
print(a8)

print("--------输出矩阵中每行或者每列的最大值-------")
data = np.sin(np.arange(20)).reshape(5, 4)
print(data)
ind = data.argmax(axis=0)  # 输出data数组每列的最大值的索引
print(ind)
# data.shape[1]用于输出矩阵的列数
print(data.shape[1])  # shape属性用于输出矩阵的行数和列数，即(5, 4)
data_max = data[ind, range(data.shape[1])]  # 输出data矩阵每列的最大值
print(data_max)

print("--------对矩阵中的元素进行复制------------")
b1 = np.arange(0, 40, 10)
print(b1)
b1.shape = (2, 2)
b2 = np.tile(b1, (4, 2))  # 将矩阵b1的行复制为原来的4倍，列复制为原来的2倍
print(b2)

print("--------------对矩阵进行排序-------------")
c = np.array([[4, 3, 5], [1, 2, 1]])
print("-------原本的矩阵为:---------")
print(c)
d = np.sort(c, axis=1)
print("------------按行排序过后的矩阵为：--------------")
print(d)
# print(help(np.argsort))
print("------np.argsort()函数表示输出排序好的元素的索引值-------")
print(np.argsort(c, axis=0))  # axis=0表示按照列进行排序
