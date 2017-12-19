# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-7HeatMap.py —— 绘制热度图
@time: 2017-12-19 16:23  
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# uniform_data = np.random.rand(3, 3)  # 更加均匀分布生成数据
# print(uniform_data)
# heat_map = sns.heatmap(uniform_data)
# 设置调色板的范围为0.2~0.5，不在这个范围内的值的颜色统一为一个颜色
# heat_map = sns.heatmap(uniform_data, vmin=0.2, vmax=0.5)

# normal_data = np.random.randn(3, 3)  # 根据标准正态分布生成数据
# print(normal_data)
# 设置调色板的center值
# heat_map = sns.heatmap(normal_data, center=0)

flights = sns.load_dataset("flights")
print(flights.head())

# 生成一个列表示年，行表示月份的矩阵，矩阵中的值表示在某一年某一月的乘客数量
flights = flights.pivot("month", "year", "passengers")
print(flights)
# heat_map = sns.heatmap(flights)  # 根据构造好的数据生成热度图
# annot=True表示在热度图的每个方格中显示出具体的数值，fmt="d"表示数值的显示格式
# heat_map = sns.heatmap(flights, annot=True, fmt="d")

# 每个格子之间添加一定的空隙
# heat_map = sns.heatmap(flights, linewidths=0.5)

# 指定color bar的颜色
# heat_map = sns.heatmap(flights, cmap="YlGnBu")

# 隐藏color bar
heat_map = sns.heatmap(flights, cbar=False)

plt.show()

