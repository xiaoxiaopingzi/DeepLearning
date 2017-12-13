# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: pandas_2.py 
@time: 2017-12-11 16:17  
"""
import pandas as pd
import numpy as np

# 读取泰坦尼克号上的幸存者数据
titanic = pd.read_csv("train.csv")

# 求出每种票的平均价格
passenger_classes = [1, 2, 3]
fares_by_class = {}  # {}表示一个空的dict
for this_class in passenger_classes:
    pclass_rows = titanic[titanic["Pclass"] == this_class]
    pclass_fares = pclass_rows["Fare"]
    fare_for_class = pclass_fares.mean()
    fares_by_class[this_class] = fare_for_class  # 向dict中添加元素，添加键值对
print(fares_by_class)

# pivot_table()表示数据透视表，index参数表示统计的基准，values表示需要统计的东西，aggfunc表示统计的方式
# 下面语句的意思是：根据船舱的等级来统计乘客获救的几率
passenger_survival = titanic.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
print(passenger_survival)  # 和前面的代码相似

# 根据船舱的等级来统计乘客的年龄
passenger_age = titanic.pivot_table(index="Pclass", values="Age")  # 默认按照平均值的方式统计
print(passenger_age)

# 根据登船地点来统计乘客的票价和存活人数，统计多个指标
# Embarked——登船地点, Fare——票价
port_stats = titanic.pivot_table(index="Embarked", values=["Fare", "Survived"], aggfunc=np.sum)
print(port_stats)

# 去除掉含有缺失值的行或者列
# specifying axis=1 or axis="columns" will drop any columns that have null values
drop_na_columns = titanic.dropna(axis=1)  # 将有缺失值的列去掉
# 将有缺失值的行去掉，指定去掉的列为Age列以及Sex列
new_titanic_survival = titanic.dropna(axis=0, subset=["Age", "Sex"])
# print(new_titanic_survival)

# 定位某一个样本的某一个特征
row_index_83_age = titanic.loc[83, "Age"]
row_index_835_pclass = titanic.loc[835, "Pclass"]
print("定位索引为83的样本的年龄:{age}".format(age=row_index_83_age))
print("索引为835的乘客的船舱类型为:{pclass}".format(pclass=row_index_835_pclass))
print("索引为835的乘客的船舱类型为:{}".format(row_index_835_pclass))
# print(help(str.format))

print("----------将数据按照Age从大到小进行排序：------")
new_titanic = titanic.sort_values("Age", ascending=False)
print(new_titanic[0:10])
titanic_reindexed = new_titanic.reset_index(drop=True)
print("--------将排序好的数据重新定义索引：-------------")
print(titanic_reindexed.loc[0:10])

# apply(func, axis=0):
#   参数：
#     func : function
#         Function to apply to each column/row
#     axis : {0 or 'index', 1 or 'columns'}, default 0
#         * 0 or 'index': apply function to each column
#         * 1 or 'columns': apply function to each row
# 这个函数用于返回索引为99的样本
def hundredth_row(column):
    hundredth_item = column.loc[99]
    return hundredth_item

# apply函数用于增加代码的可读性
print(titanic.apply(hundredth_row))

def not_null_count(column):
    column_null = pd.isnull(column)  # 判断数据矩阵中是否存在缺失值，如果存在就输出True
    # print(column_null)
    null = column[column_null]
    # print(null)
    return len(null)

print(titanic.apply(not_null_count))

def which_class(row):
    pclass = row['Pclass']
    if pd.isnull(pclass):
        return "Unknown"
    elif pclass == 1:
        return "First Class"
    elif pclass == 2:
        return "Second Class"
    elif pclass == 3:
        return "Third Class"

print(titanic.apply(which_class, axis=1))
# print(help(titanic.apply))