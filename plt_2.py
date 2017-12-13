# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: plt_2.py 
@time: 2017-12-13 15:19  
"""
import matplotlib.pyplot as plt
import numpy as np

# 画子图
fig = plt.figure(figsize=(6, 7))
x = np.linspace(0, 10, 100)
# 子图的矩阵的大小为2 X 2
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(np.random.randint(1, 5, 5), np.arange(5), color="blue")
ax2.plot(x, np.sin(x),  label="$sin$", color="blue", linewidth=2)
ax2.plot(x, np.cos(x),  label="$cos$", color="red", linewidth=2)

for key, spine in ax2.spines.items():
    spine.set_visible(False)  # 去除子图的方框

# 设置子图的x轴和y轴的范围
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(0, 10)
ax2.set_ylim(-1, 1)
ax2.set_title("y = sin or cos")

ax2.text(0, 1, "cos")
ax2.text(0, 0, "sin")

# tick_params()函数用于将图中坐标轴的短线去掉
plt.tick_params(bottom="off", top="off", left="off", right="off")

plt.show()


