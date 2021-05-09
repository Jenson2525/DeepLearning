#机器学习应用：鸢尾花分类
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()  #加载数据集
print(iris_dataset['DESCR'][:193] + "\n...")  #数据集的说明
print("Target names: {}".format(iris_dataset['target_names']))  #要预测的花的品种
print("Feature names: {}".format(iris_dataset['feature_names']))  #对每一个特征进行说明

#衡量模型是否成功
#打乱数据集并进行拆分
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
#机器学习模型
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x_train, y_train)
x_new = np.array([[5, 2.9, 1, 0.2]])
print("x_new.shape: {}".format(x_new.shape))
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Prediction target name : {}".format(iris_dataset['target_names'][prediction]))
#评估模型
y_pred = knn.predict(x_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))