# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: xgboost.py 
@time: 2017-12-25 22:14

xgboost集成学习方法
"""
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

# 装载数据
data = loadtxt("pima-indians-diabetes.csv", delimiter=",")

# 切分数据为x和y
X = data[:, 0:8]
Y = data[:, 8]
# 切分数据为训练集和测试集
sees = 7  # 随机种子
test_size = 0.33  # 测试集的大小为33%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=sees)

# 训练模型
# xgboost中的一些参数:
#   1、learning rate
#   2、二叉树的一些参数
#       max_depth —— 树的最大深度
#       min_child_weight —— 叶子节点的最小权重
#       subsample —— 随机森林中随机选择一部分样本, colsample_bytree——随机森林中随机选择一部分特征
#       gamma —— 叶子节点个数前面的惩罚项
#   3、正则化参数
#       lambda
#       alpha
xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5,
                     min_child_weight=1, gamma=0, subsample=0.8,
                     colsample_bytree=0.8, objective="binary:logistic",
                     nthread=4, scale_pos_weight=1, seed=27)
model = XGBClassifier()
# model.fit(X_train, y_train)
eval_set = [(X_test, y_test)]
# Will train until validation_0-logloss hasn't improved in 10 rounds.
# verbose=True表示打印出每加入一棵树后logloss的值
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_set,
          eval_metric="logloss", verbose=True)

# 针对测试数据做出决策
y_pred = model.predict(X_test)
# round()函数的作用是对一个小数进行四舍五入，保留小数点后一位
predictions = [round(value) for value in y_pred]
# print(predictions)

# 评估预测值的精度
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:%.3f%%" % (accuracy * 100.0))

# 显示出每个特征的重要性
plot_importance(model)
pyplot.show()
