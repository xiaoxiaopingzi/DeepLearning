# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: RandomForest.py —— 决策树案例
@time: 2017-12-20 10:35  
"""
import seaborn as sns
import matplotlib.pyplot as plt

# 从https://github.com/mwaskom/seaborn-data获取数据集
iris = sns.load_dataset("iris")  # 鸢尾花数据集
print(iris.head())
print(iris.describe())

# 画出每两个特征之间的关系
# sns.pairplot(iris, hue="species")
# 传进来的数据不能有缺失值，如果有缺失值则需要去掉缺失值
# sns.pairplot(iris.dropna(), hue="species")

# 画出鸢尾花的种类和各个特征之间的关系
plt.figure(figsize=(10, 10))
for column_index, column in enumerate(iris.columns):
    if column == "species":
        continue
    else:
        plt.subplot(2, 2, column_index + 1)
        sns.violinplot(x="species", y=column, data=iris)

plt.show()
