# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-4Logic.py —— 线性回归分析绘图
@time: 2017-12-18 20:18  
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
print(tips.head())

# 可以使用regplot()和lmplot()函数，两者的功能差不多
# sns.regplot(x="total_bill", y="tip", data=tips)
# sns.lmplot(x="total_bill", y="tip", data=tips)

# 给x值加上一些微小的抖动
sns.regplot(x="size", y="tip", data=tips, x_jitter=0.05)

plt.show()

