# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-Style.py —— 设置主题风格
@time: 2017-12-18 10:03

Seaborn库是对matplotlib库的一个封装，给我们提供了非常丰富的模板
5中主题风格：
 darkgrid
 whitegrid
 dark
 white
 ticks
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

# sns.set_style("dark")
# sns.set_style("white")
# sns.set_style("ticks")
# sinplot()
# sns.despine()  # 去掉图中上方和右方的轴
# sns.despine(offset=10)  # offset参数表示图离x，y坐标轴的距离


# sns.set_style("whitegrid")
# data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
# sns.boxplot(data=data)
# sns.despine(left=True)  # left=True表示在去掉图中上方和右方的轴的基础上，隐藏左边的坐标轴

# 使用with语句，使得sns设置的风格只在with语句的子图中起作用
# 这样便于子图的对比
# with sns.axes_style("darkgrid"):
#     plt.subplot(211)
#     sinplot()
# plt.subplot(212)
# sinplot(-1)


sns.set()  # 使用sns默认的设置
# sns.set_context("paper")
# sns.set_context("poster")
# sns.set_context("talk")
# font_scale表示设置坐标轴字体的大小，rc={"line.linewidth": 2.5}表示设置线宽
sns.set_context("notebook", font_scale=1, rc={"line.linewidth": 2.5})
# plt.figure(figsize=(8, 6))
sinplot()

plt.show()

