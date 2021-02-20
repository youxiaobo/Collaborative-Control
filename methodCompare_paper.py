#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.preprocessing import label_binarize





# 1.读入数据
file = open('.\\input_complete.csv')
data = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# 2.划分输入和输出
array_data = data.values
input_x = array_data[..., 0:14]#前14列为输入
output_y = array_data[..., 14:]#第15列为输出

# 对输入进行独热编码
ohe_x = OneHotEncoder(sparse=True, categories='auto')
ohe_x.fit(input_x)
# print(ohe.active_features_)
# print(ohe.feature_indices_)
# print(ohe.n_values_)
input_x = ohe_x.transform(input_x)
# print(input_x[:2, :])
# print(input_x[:2, :].todense())



# 3. 随机划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(input_x, output_y, random_state=0, train_size=0.7, test_size=0.3)


y_train_hot = y_train - 1
y_train_hot = np_utils.to_categorical(y_train_hot, num_classes=10)
y_test_hot = y_test - 1
y_test_hot = np_utils.to_categorical(y_test_hot, num_classes=10)

# 方法一 SVM
# SVM对核函数的选择具有不敏感性
# 其中c表示对误差的惩罚程度，c越大，表示对误差的惩罚程度越大，模型对样本数据的学习更精确，也因此容易过拟合；
# 反之，c越小，对误差的惩罚程度越小，可能欠拟合。
# g对应RBF核中的gamma值，gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，也就是过拟合；
# 反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，也就是欠拟合。
model_svm = SVC(C=10, kernel='rbf', gamma=0.1, decision_function_shape='ovr', probability=True)

model_svm.fit(x_train, y_train)
print('SVM训练集准确率：%0.3f' % model_svm.score(x_train, y_train))
print('SVM测试集准确率：%0.3f' % model_svm.score(x_test, y_test))


svm_score = model_svm.predict_proba(x_test)
print('调用函数auc：', metrics.roc_auc_score(y_test_hot, svm_score, average='micro'))
svm_fpr, svm_tpr, svm_thresholds = metrics.roc_curve(y_test_hot.ravel(), svm_score.ravel())
svm_auc = metrics.auc(svm_fpr, svm_tpr)
print('手动计算auc：', svm_auc)


# 方法二 决策树、随机森林


model_tree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=9)
model_tree.fit(x_train, y_train)
print("DecisionTree训练集准确率:%0.3f" % model_tree.score(x_train, y_train))
print("DecisionTree测试集准确率:%0.3f " % model_tree.score(x_test, y_test))

tree_score = model_tree.predict_proba(x_test)
print('调用函数auc：', metrics.roc_auc_score(y_test_hot, tree_score, average='micro'))
tree_fpr, tree_tpr, tree_thresholds = metrics.roc_curve(y_test_hot.ravel(),tree_score.ravel())
tree_auc = metrics.auc(tree_fpr, tree_tpr)
print ('手动计算auc：', tree_auc)


# 方法三 神经网络
# 为了得到可重复的实验结果，设置随机数种子
seed = 7
np.random.seed(seed)

# 最后选择的参数
model_NN = Sequential([Dense(50, input_dim=36), Activation('relu'),
        Dense(20), Activation('relu'),
        Dense(10), Activation('softmax')])
model_NN.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_NN.fit(x_train, y_train_hot, epochs=100, batch_size=30, verbose=0)
loss, accuracy = model_NN.evaluate(x_test, y_test_hot)
#print('NN 训练集准确率: %0.3f 损失率：%0.3f' % (history.history.get('acc')[99], history.history.get('loss')[99]))
print('NN 训练集准确率: %0.3f 损失率：%0.3f' % (history.history['accuracy'][99], history.history['loss'][99]))
print('NN 测试集准确率: %0.3f 损失率：%0.3f' % (accuracy, loss))
model_NN.summary()

model_NN.save(".\\nn_model.h5")

NN_score = model_NN.predict_proba(x_test)
print('调用函数auc：', metrics.roc_auc_score(y_test_hot, NN_score, average='micro'))
NN_fpr, NN_tpr, NN_thresholds = metrics.roc_curve(y_test_hot.ravel(), NN_score.ravel())
NN_auc = metrics.auc(NN_fpr, NN_tpr)
print('手动计算auc：', NN_auc)


# 将三条ROC曲线绘制在同一张图上
plt.figure()
plt.plot(svm_fpr, svm_tpr, c='b', lw=2, alpha=0.7, label=u'SVM AUC=%.3f' % svm_auc)
plt.plot(tree_fpr, tree_tpr, c='y', lw=2, alpha=0.7, label=u'tree AUC=%.3f' % tree_auc)
plt.plot(NN_fpr, NN_tpr, c='r', lw=2, alpha=0.7, label=u'NN AUC=%.3f' % NN_auc)
plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)


plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'ROC curve comparison', fontsize=17)
plt.show()
