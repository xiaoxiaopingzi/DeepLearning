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
