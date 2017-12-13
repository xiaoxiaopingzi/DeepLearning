# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: plt_1.py 
@time: 2017-12-13 11:03
画折线图
"""
import matplotlib.pyplot as plt
import numpy as np

# 画折线图
x = np.linspace(0, 10, 100)
fig = plt.figure(figsize=(6, 6))

# 自定义线条的颜色
cb_dark_blue = (0/255, 107/255, 164/255)

# 用label标签来指定图中每条线所代表的含义,加上$$符号会使文字有一点倾斜
plt.plot(x, np.sin(x), label="$sin$", color=cb_dark_blue, linewidth=2)
plt.plot(x, np.cos(x), label="$cos$", color="red", linewidth=2)

# loc参数用于指定label标签在图中的位置
# loc : int or string or pair of floats, default: 'upper right'
#         The location of the legend. Possible codes are:
plt.legend(loc="best")  # 只有加上这条语句，label标签才会显示出来
# print(help(plt.legend))

plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0.0, 9.0)
plt.ylim(-1.0, 1.0)
plt.title("y = sin(x) or cos(x)")

# 在x=0，y=0添加sin标记，在x=0，y=1添加cos标记
plt.text(0, 0, "sin")
plt.text(0, 1, "cos")

# tick_params()函数用于将图中坐标轴的短线去掉
plt.tick_params(bottom="off", top="off", left="off", right="off")

plt.xticks(rotation=45)  # 使x轴上坐标轴的标号向左旋转一定的角度
plt.yticks(rotation=45)  # 使y轴上坐标轴的标号向右旋转一定的角度

plt.show()
