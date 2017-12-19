# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-5Multivariate.py —— 多变量分析绘图
@time: 2017-12-18 20:48  
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 当x为离散值且数据较多时，数据容易重叠
# sns.stripplot(x="day", y="total_bill", data=tips)
# 给数据加上一点抖动
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)

# swarmplot()函数会自动原有数据的基础上加上一些抖动，并且这些抖动不是随机的，会使数据均匀分布在原始数据的两边
# sns.swarmplot(x="day", y="total_bill", data=tips)

# 加上hue="sex"参数表示将图中的数据按照sex的值绘制不同的颜色(即将数据按照sex参数进行划分)
# sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)

# 将x和y的位置调换，就是将上图横着画
# sns.swarmplot(x="total_bill", y="day", hue="sex", data=tips)


# 画盒图——先将数据从小打大进行排列，然后选取位于1/4、1/2、3/4位置的元素的值作为三个分位点
# 箱体的左侧(下)边界代表第一四分位(Q1),而右侧(上)边界代表第三四分位(Q3)
# IQR即统计学中四分位距，表示第1/4分位和3/4位之间的距离(IQR = Q3-Q1，即上四分位数与下四分位数之间的差，也就是盒子的长度。)
# 最小观测值为min = Q1 - 1.5*IQR, 最大观测值为max = Q3 + 1.5*IQR
# 如果一个点的值大于最大观测值或者小于最小观测值，则该点为离群点
# sns.boxplot(x="day", y="total_bill", hue="time", data=tips)  # 将数据按照time参数进行划分
# sns.boxplot(x="day", y="total_bill", data=tips)

# 画出一张类似小提琴的图,数据越多的地方越粗，数据越少的地方越细
# sns.violinplot(x="day", y="total_bill", hue="time", data=tips)
# 添加split=True属性，这种情况下，以sex属性的值为分割条件，左边为male，右边为female
# sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)

# 将violinplot图和swarmplot图画在一起
# sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
# sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=0.5)  # alpha参数表示透明程度

# 显示值的集中趋势，可以使用条形图
# titanic = sns.load_dataset("titanic")
# sns.barplot(x="sex", y="survived", hue="class", data=titanic)

# 点图可以更好的描述变化的差异性
# sns.pointplot(x="sex", y="survived", hue="class", data=titanic)

# 点图的第二个例子,g表示green
# sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
#               palette={"male": "g", "female": "m"}, markers=["^", "o"],
#               linestyles=["-", "--"])

# 宽型数据
# iris = sns.load_dataset("iris")
# sns.boxplot(data=iris, orient="h")  # 加上h参数用于横着画盒图

# 多层面板分类图 —— 相当于将上面的一些图进行了整合
# kind : {``point``, ``bar``, ``count``, ``box``, ``violin``, ``strip``, "swarm"}
# 默认是点图
# sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)
# 画条形图
# sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")
# 画散点图，col="time"表示按照time的值的多少画多个图
# sns.factorplot(x="day", y="total_bill", hue="smoker", col="time", data=tips, kind="swarm")
# 画盒图，size=4, aspect=0.5表示设定图的大小和长宽比
sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=0.5)

# print(help(sns.factorplot))
plt.show()
