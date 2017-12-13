# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: pandas_3.py 
@time: 2017-12-12 14:59  
"""
import pandas as pd
from pandas import Series
import numpy as np

# Series类型是DataFrame类型的子集
titanic = pd.read_csv("train.csv")
series_age = titanic["Age"]
print(type(series_age))  # <class 'pandas.core.series.Series'>
print(series_age[0:6])
series_name = titanic["Name"]
print(series_name[0:5])

# 自己构造一个Series
name = series_name.values  # 获取series_name中所有的值
age = series_age.values     # 获取series_age中所有的值
# 以人名为索引，年龄为值构造一个Series
series_custom = Series(age, index=name)
# 取出索引为Braund, Mr. Owen Harris以及Allen, Mr. William Henry的Series
# 输出结果为：
#   Braund, Mr. Owen Harris     22.0
#   Allen, Mr. William Henry    35.0
print("--------将人名作为索引，取出下面的数据：-------")
print(series_custom[["Braund, Mr. Owen Harris", "Allen, Mr. William Henry"]])
print("----------以数字作为索引，取出下面的数据:--------------")
print(series_custom[0:4])

print("------对人名的索引进行排序，按照排序好的索引对series_custom进行排序：--------")
original_index = series_custom.index.tolist()
sorted_index = sorted(original_index)
sorted_by_index = series_custom.reindex(sorted_index)
print(sorted_by_index)
print("----------------------------")

# 直接使用sort_values()函数和sort_index()函数对series进行排序
sc2 = series_custom.sort_values()
sc3 = series_custom.sort_index()
print(sc2[100:110])
print("------------------------------")
print(sc3[0:10])

# series可以用在numpy中，即series合numpy中的ndarray类型是一样的
print(np.add(series_custom, series_custom))
# print(np.sin(series_custom))
print(np.max(series_custom))
