# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: plt_4.py 
@time: 2017-12-13 20:36
画直方图和盒图
"""
import numpy as np
import matplotlib.pyplot as plt

# 概率分布直方图
# 本例是标准正态分布
figure1 = plt.figure(1)
# 设置均值，中心所在点
mean = 2
# 用于将每个点都扩大响应的倍数
sigma = 2

# x中的点分布在 mean 旁边，以mean为中点
# randn()函数的作用：Return a sample (or samples) from the "standard normal" distribution.
x = mean + sigma * np.random.randn(1000)
# print(help(np.random.randn))

# bins 设置分组的个数
# normed 是否对y轴数据进行标准化(如果为True，则表示在本区间的点在所有的点中所占的频率)
# 如果 normed 为False， 则是显示点的数量
# plt.hist(x, bins=30, color="red", normed=True)
plt.hist(x, bins=10, normed=False, range=(2, 7))
plt.show()



# 画盒图
figure2 = plt.figure(2)
matrix = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
print(matrix)

# 画盒图，矩阵的每一列就是一个盒图
plt.boxplot(matrix)

# xticks函数用于设置x轴上标签的显示内容以及旋转角度
# xticks( arange(12), calendar.month_name[1:13], rotation=17 )
# help(plt.xticks)
# 下面的语句表示x轴有四个标签，并且每个标签要旋转17度
name = ["wanzhi", "chen", "huang", "luo"]
plt.xticks(np.arange(4) + 1, name, rotation=17)

plt.title("Box diagram")
plt.xlabel("this is x")
plt.ylabel("this is y")
plt.show()
