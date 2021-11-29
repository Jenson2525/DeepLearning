from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

#导入数据
digits = load_digits()
#数据分割
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)

#对训练和测试的数据集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#模型训练
lsvc = LinearSVC()  #初始化
lsvc.fit(X_train, y_train)  #模型训练
y_predict = lsvc.predict(X_test)  #利用训练好的模型对测试样本进行预测

#模型评估
print('The Accuracy of Linear SVC: ', lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names = digits.target_names.astype(str)))
