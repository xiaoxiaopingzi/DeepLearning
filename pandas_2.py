# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: pandas_2.py 
@time: 2017-12-11 16:17  
"""
import pandas as pd

death_valley_2014 = pd.read_csv("death_valley_2014.csv")
max_TemperatureF = death_valley_2014["Max TemperatureF"]

# 判断某一行是否有缺失值，有缺失值就为True，没有就为False
isNull = pd.isnull(max_TemperatureF)
print(isNull)
# 只保留缺失值
max_TemperatureF_null = max_TemperatureF[isNull]
print(max_TemperatureF_null)  # 打印出缺失值
print(len(max_TemperatureF_null))  # 打印出缺失值的个数

# 如果不处理缺失值，就无法得到所有数据的平均值
mean_T = sum(max_TemperatureF) / len(max_TemperatureF)
print(mean_T)

print("过滤掉缺失值后的平均值为：")
# 利用Pandas中的过滤将缺失值去掉
good_TemperatureF = max_TemperatureF[isNull == False]  # 去掉缺失值
correct_mean_T = sum(good_TemperatureF) / len(good_TemperatureF)
print(correct_mean_T)

# pandas中的mean()方法会自动去掉缺失值(pandas中的许多方法都会自动过滤缺失值)
print(max_TemperatureF.mean())
