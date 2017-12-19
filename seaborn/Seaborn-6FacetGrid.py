# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-6FacetGrid.py —— FacetGrid使用方法
@time: 2017-12-19 14:45  
"""
import seaborn as sns
from pandas import Categorical
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
print(tips.head())

# 首先实例化一个FacetGrid出来，col="time"表示根据time的值画出多个子图
# g = sns.FacetGrid(tips, col="time")
# 通过g.map()函数来画出实际的图
# g.map(plt.hist, "tip")

# 实例化一个FacetGrid对象，根据sex画出两个子图，每个子图用是否是smoker来区别两种点
# g = sns.FacetGrid(tips, col="sex", hue="smoker")
# g.map(plt.scatter, "total_bill", "tip", alpha=0.7)  # alpha表示透明度
# g.add_legend()   # 将图中的解释框显示出来

# row="smoker", col="time"表示根据时间和是否抽烟画出四个子图
# g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
# 画线性回归图，fit_reg=True表示需要画出回归的直线
# g.map(sns.regplot, "size", "total_bill", color="0.1", fit_reg=True, x_jitter=0.1)

# 需要画出什么东西，用什么数据，图的大小和长宽比是多少，得到一个空的图
# g = sns.FacetGrid(tips, col="day", size=4, aspect=0.5)
# g.map(sns.barplot, "sex", "total_bill")  # 在上一步生成的空图中，实际的填充数据

# 指定顺序
# ordered_days = tips.day.value_counts().index
# print(ordered_days)
# sns中的数据格式是pandas中的DataFrame格式，因此指定的顺序的格式需要是pandas中的数据格式
# ordered_days = Categorical(["Thur", "Fri", "Sat", "Sun"])
# g = sns.FacetGrid(tips, row="day", row_order=ordered_days)
# g.map(sns.boxplot, "total_bill")

# 设定颜色
# pal = dict(Lunch="seagreen", Dinner="gray")
# g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
# g.map(plt.scatter, "total_bill", "tip", s=50, alpha=0.7,
#       linewidth=0.5, edgecolor="white")  # s表示图中点的大小
# g.add_legend()

# 指定标记类型，^表示采用上三角形，v表示采用下三角形
g = sns.FacetGrid(tips, hue="sex", palette="Set1",
                  size=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=80,
      linewidth=0.5, edgecolor="white")
g.add_legend()

plt.show()
