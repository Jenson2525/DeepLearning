# 文件功能：knn实现鸢尾花数据集分类
from sklearn import datasets                           # 引入sklearn包含的众多数据集
from sklearn.model_selection import train_test_split   # 将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier     # 利用knn方式训练数据
 
# 【1】引入训练数据
iris = datasets.load_iris() # 引入iris鸢尾花数据,iris数据包含4个特征变量
iris_X = iris.data          # 特征变量
iris_y = iris.target        # 目标值
# 利用train_test_split进行将训练集和测试集进行分开，test_size占30%
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.8)
print(y_train)              #训练数据的特征值分为3类
 
# 【2】执行训练
knn = KNeighborsClassifier()   # 引入训练方法
knn.fit(X_train, y_train)      # 进行填充测试数据进行训练
  
# 【3】预测数据
print(knn.predict(X_test))      # 预测特征值
print(y_test)                   # 真实特征值
 
# 【4】可直接调用accuracy_score计算准确率  
from sklearn.metrics import accuracy_score
print("测试准确度：", accuracy_score(knn.predict(X_test), y_test))