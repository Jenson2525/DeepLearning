import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

#数据预处理
#创建特征列表
columnNames = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 
               'Uniformity of Cell Shape', 'Magrginal Adhesion', 'Single Epithelial Cell Szie',
               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoil', 'Mitoses', 'Class']
#导入数据并处理
data = pd.read_csv('D:\\VSCode\\DeepLearning\\BreastCancer\\breast-cancer-wisconsin.data', names = columnNames)
data = data.replace(to_replace = '?', value = np.nan)  #将?替换为标准缺失值表示
data = data.dropna(how = 'any')  #删除带有缺失值的数据
#准备训练数据、测试数据
X_train, X_test, y_train, y_test = train_test_split(data[columnNames[1:10]], data[columnNames[10]], 
                                                    test_size = 0.25, random_state = 33)

#使用线性分类模型进行良性/恶性肿瘤预测
#标准化数据
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#训练模型及分类
#使用LogisticRegression进行训练和预测
lr = LogisticRegression()
lr.fit(X_train, y_train)  #使用fit函数用来训练模型参数
lr_y_predict = lr.predict(X_test)  #使用训练好的模型进行预测
print('ACC of LR: ', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Malignant']))
#使用SGDClassifier进行训练和预测
sgdc = SGDClassifier()
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)
print('ACC of SGDC: ', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names = ['Benign', 'Malignant']))
