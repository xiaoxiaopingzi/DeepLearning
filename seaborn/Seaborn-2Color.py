# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: Seaborn-2Color.py —— 设置颜色
@time: 2017-12-18 10:52  
"""
import seaborn as sns
import matplotlib.pyplot as plt

# 输出默认的分类色板
# current_palette = sns.color_palette()
# sns.palplot(current_palette)

# 输出8种颜色的分类色板
# sns.palplot(sns.color_palette("hls", 8))

# 使用hls_palette()函数来控制颜色的亮度和饱和度
#   l - 亮度(lightness)
#   s - 饱和度(saturation)
# sns.palplot(sns.hls_palette(8, l=0.7, s=0.7))

# 使用Paired关键字来输出成对的颜色
# 本例中8表示输出4对颜色，同一对颜色的深浅不同，不同对颜色的颜色不同
# sns.palplot(sns.color_palette("Paired", 8))

# 在实际的画图中使用8中颜色的色板
# sns.set_style("whitegrid")
# data = np.random.normal(size=(20, 8)) + np.arange(8) / 2
# sns.boxplot(data=data, palette=sns.color_palette("hls", 8))


# 使用xkcd来命名颜色
# xkcd中包含了一套针对随机RGB颜色的命名，产生了954个可以随时从xkcd_rgb字典中调用的已经被命名的颜色
# plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)  # lw表示线宽
# plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
# plt.plot([0, 1], [0, 3], sns.xkcd_rgb["denim blue"], lw=3)

# colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
# sns.palplot(sns.xkcd_palette(colors))

# -------------------------------------------------------------------------------

# 连续色板——色彩随数据变换，比如数据越重要则颜色就越深
# sns.palplot(sns.color_palette("Blues"))

# 如果想要翻转渐变，可以在面板名称中添加一个_r后缀
# sns.palplot(sns.color_palette("BuGn_r"))

# 色调线性变化
# 颜色的亮度和饱和度呈线性变化
# sns.palplot(sns.color_palette("cubehelix", 8))
# sns.palplot(sns.cubehelix_palette(8, start=0.5, rot=-0.75))

# light_palette()和dark_palette()调用定制连续调色板
# sns.palplot(sns.light_palette("green"))
# sns.palplot(sns.dark_palette("purple"))  # 颜色由浅到深变化
# sns.palplot(sns.light_palette("purple", reverse=True))  # 颜色由深到浅变化

sns.palplot(sns.light_palette((210, 90, 60), input="husl"))


plt.show()
