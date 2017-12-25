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
print(matrix.shape)  # 输出矩阵的行数和列数，即2 x 3
print(matrix[:, 1])  # 用冒号占位符表示取所有元素，这里表示取矩阵第1列的所有元素
print(matrix[:, 0:2])  # 取矩阵第0列到第1列的所有元素(包括头不包括尾)

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
print(matrix.sum(axis=1))  # 对矩阵按行求和
print(matrix.sum(axis=0))  # 对矩阵按列求和

# 矩阵的一些属性
# arange()函数用于生成一个array数组
print(np.arange(15))
a = np.arange(15).reshape(3, 5)  # reshape(3, 5)函数将行向量变换成3行5列的矩阵
print(a)
print(a.shape)  # shape属性用于输出矩阵的行数和列数
print(a.ndim)  # ndim属性用于输出矩阵的维数(一般是二维的)
print(a.dtype.name)
print(a.size)  # 输出矩阵中元素的个数

# 矩阵初始化的一些方法
print(np.zeros((3, 4)))  # 输出一个3行4列的初始参数，注意传入的参数是(3, 4)，即元组类型的
print(np.ones((3, 5), dtype=np.int32))  # 数据类型需要是numpy支持的类型
print(np.arange(10, 30, 5))  # 输出从10开始，到30结束(不包括30)，间隔为5的数组
print(np.arange(0, 2, 0.3))
print(np.random.random((2, 3)))  # Return random floats in the half-open interval [0.0, 1.0)
# print(help(np.random.random))

print(np.linspace(0, 2 * np.pi, 10))  # 在0到2*pi之间平均找到10个数
print(np.linspace(0, 2, 5))  # 生成5个数的数组，数组的开头和结尾分别为0.0和2.0，每两个数之间的差值为(2-0)/(5-1)
print(np.linspace(0, 4, 10, endpoint=False))  # endpoint=False表示不包括4
# print(help(np.linspace))


# numpy中矩阵的加减操作
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(a)
print(b)
print(a - b)
print(a - 1)  # python中的广播会将数字1变为一个全1的数组
print(b ** 2)  # 输出矩阵中每个元素的平方值
print(a < 35)  # 判断矩阵中每个元素的值是否小于35，是就输出True，否就输出False

print("----------")
# numpy中矩阵的乘法
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
print("矩阵A为：")
print(A)
print("----------")
print("矩阵B为：")
print(B)
print("----------")
print("矩阵A和B的内积(对应位置元素相乘)为：")
print(A*B)  # A*B表示矩阵A和矩阵B的内积——对应坐标元素相乘
print("-------------")
print("矩阵A和B相乘的结果为：")
print(A.dot(B))
print("------------")
print(np.dot(A, B))
