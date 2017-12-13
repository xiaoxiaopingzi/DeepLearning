# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: plt_3.py 
@time: 2017-12-13 19:39  
画散点图和柱状图
"""

import matplotlib.pyplot as plt
import numpy as np

# 画柱状图
figure = plt.figure(1)
bar_positions = np.arange(5) + 0.75
bar_heights = np.arange(5) + 5 * np.random.random() + 0.01
print("bar_heights:{}".format(bar_heights))
print("bar_positions:{}".format(bar_positions))
# 画柱状图，0.3表示柱的宽度
plt.bar(bar_positions, bar_heights, 0.3, color="blue", label="$Vertical$")

# 横着画柱状图
plt.barh(bar_positions, bar_heights, 0.3, color="red", label="$Horizontal$")

plt.title("random")
plt.xlabel("this is x")
plt.ylabel("this is y")

plt.xticks(rotation=45)
plt.legend(loc="best")
plt.show()

# 画散点图
figure2 = plt.figure(2)
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.scatter(x, y)

plt.title("y = sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.show()
