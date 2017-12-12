# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: pandas_1.py 
@time: 2017-12-11 10:09  
"""
import pandas

death_valley_2014 = pandas.read_csv("death_valley_2014.csv")
# pandas中的数据类型为：<class 'pandas.core.frame.DataFrame'>
print(type(death_valley_2014))

# 输出csv文件中每列数据的数据类型
# object - For String values
# int  - For integer values
# float - For float values
# datetime - For time values
# bool - For Boolean values
print(death_valley_2014.dtypes)
# print(help((pandas.read_csv)))

# 打印出csv文件中头3行数据
print(death_valley_2014.head(3))
print(death_valley_2014.tail(4))  # 打印出csv文件中最后4行的数据

# 将csv文件每列的列名打印出来(其实就是将csv文件第一行的数据打印出来)
print(death_valley_2014.columns)

print(death_valley_2014.shape)  # 输出csv文件中数据的行数和列数

# 打印出指定索引的数据，注意索引不能越界
print(death_valley_2014.loc[0])  # 打印出第0行数据
print(death_valley_2014.loc[7])  # 打印出第7行数据
print(death_valley_2014.loc[3:6])  # 打印出第3行到第6行的数据(包括第6行)
print(death_valley_2014.loc[[0, 3]])  # 打印出第0行和第3行的数据

# 打印出某一列的所有数据
print(death_valley_2014["Max TemperatureF"])
# 打印出某两列的数据
print(death_valley_2014[["Max TemperatureF", "Mean TemperatureF"]])

# 找到列名中最后一个字符是F的列的数据
col_name = death_valley_2014.columns.tolist()
print(col_name)
gram_columns = []
for c in col_name:
    if c.endswith("F"):
        gram_columns.append(c)
gram_df = death_valley_2014[gram_columns]
print(gram_df.head(4))

# 取出某一列的最大值
print(death_valley_2014["Max TemperatureF"].max())

# 按照某一列进行从小到大排序，inplace=True表示用排序好的矩阵替换原来的矩阵，输出排序好的值
death_valley_2014.sort_values("Max TemperatureF", inplace=True)
print(death_valley_2014["Max TemperatureF"])

# 降序(从大到小)排列
death_valley_2014.sort_values("Max TemperatureF", inplace=True, ascending=False)
print(death_valley_2014["Max TemperatureF"])


# 判断某一行是否有缺失值，有缺失值就为True，没有就为False
max_TemperatureF = death_valley_2014["Max TemperatureF"]
isNull = pandas.isnull(max_TemperatureF)
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

