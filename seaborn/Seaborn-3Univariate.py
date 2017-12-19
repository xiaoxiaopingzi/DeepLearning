# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-3s.py —— 单变量分析绘图
@time: 2017-12-18 16:54  
"""
import seaborn as sns
import numpy as np
from scipy import stats, integrate
import pandas as pd
import matplotlib.pyplot as plt

# 对单个特征进行分析，采用概率分布直方图
# 画概率分布直方图
# x = np.random.normal(size=100)
# sns.distplot(x, kde=False)

# 指定bias的个数
# sns.distplot(x, bins=20, kde=False)


# 在图中画出数据分布情况, 使用fit参数在图中画出数据的变化趋势
# x2 = np.random.gamma(6, size=200)
# sns.distplot(x2, kde=False, fit=stats.gamma)


# 观察两个变量之间的关系，采用散点图
# mean, cov = [0, 1], [(1, 0.5), (0.5, 1)]

# 第一种表示方式
# data = np.random.multivariate_normal(mean, cov, 200)  # 根据均值和协方差生成数据
# df = pd.DataFrame(data, columns=["x", "y"])
# sns.jointplot(x="x", y="y", data=df)

# 散点图的第二种表示方式
# 当数据量较大时，可能会出现一个地方出现的点过多，无法清晰的看出数据的分布，这是可以使用下面的hex画图方式
# x, y = np.random.multivariate_normal(mean, cov, 1000).T
# with sns.axes_style("white"):
#     sns.jointplot(x=x, y=y, kind="hex", color="k")  # 颜色越深，表示在这个区域的点越多


# 观察四个特征之间的关系
# 使用内置的数据集
iris = sns.load_dataset("iris")
# 会得到一个4 X 4即共16个子图的图，对角线的子图表示一个特征的概率直方图
# 非对角线的子图表示两个特征之间的关系
sns.pairplot(iris)

plt.show()
