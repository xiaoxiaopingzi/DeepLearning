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
# g = sns.FacetGrid(tips, hue="sex", palette="Set1",
#                   size=5, hue_kws={"marker": ["^", "v"]})
# g.map(plt.scatter, "total_bill", "tip", s=80,
#       linewidth=0.5, edgecolor="white")
# g.add_legend()

# 指定x轴标签、x轴范围、y轴范围等
# with sns.axes_style("white"):
#     g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
# g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=0.5)
# g.set_axis_labels("Total bill (US Dollars)", "Tip")  # 设置x轴标签
# g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])  # 设置坐标轴的范围
# g.fig.subplots_adjust(wspace=0.02, hspace=0.02)  # 对子图的位置进行调整
# g.fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1,
#                       top=0.9, wspace=0.02, hspace=0.02)  # 对子图的位置进行调整

iris = sns.load_dataset("iris")
print(iris.head())

# PairGrid()函数会画出所有特征之间的关系
# g = sns.PairGrid(iris)

# g = sns.PairGrid(iris, hue="species")  # 添加一个species用于区分不同的物种

# 只采用"sepal_length"和"sepal_width"这两个特征
# g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")

# g.map(plt.scatter)  # 所有子图都采用散点图

# g.map_diag(plt.hist)   # 位于对角线的子图采用直方图
# g.map_offdiag(plt.scatter)   # 非对角线的子图采用散点图
# g.add_legend()

# 添加一个调色板
g = sns.PairGrid(tips, hue="size", palette="GnBu_d")
g.map(plt.scatter, s=50, edgecolor="white")
g.add_legend()

plt.show()
