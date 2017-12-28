# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: ARIMA_Models.py 
@time: 2017-12-27 11:20

ARIMA时间序列模型
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

data = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
        6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355,
        10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707, 10767,
        12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
        13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
        9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
        11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]

# print(len(data))
dta = pd.Series(data)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))


def tsplot(y, lags=None, title="", figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    acf_ax = plt.subplot2grid(layout, (0, 1))
    diff1_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    diff1 = y.diff(1)
    diff1.plot(ax=diff1_ax)
    diff1_ax.set_title("First order difference graph")
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


# tsplot(dta, title="Sequence data", lags=36)
# plt.show()

arima200 = sm.tsa.SARIMAX(dta, order=(7, 1, 0))
model_result = arima200.fit()
