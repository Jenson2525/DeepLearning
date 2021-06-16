# 文件功能：svm分类鸢尾花数据集
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
 
#【1】读取数据集  
data = load_iris()
 
#【2】划分数据与标签  
x = data.data[:, :2]
y = data.target
train_data, test_data, train_label, test_label = train_test_split\
                    (x, y, random_state=1, train_size=0.6, test_size=0.4)
print(train_data.shape)
 
#【3】训练svm分类器  
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo') # ovr:一对多策略  
classifier.fit(train_data, train_label.ravel()) #ravel函数在降维时默认是行序优先  
 
#【4】计算分类器的准确率  
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))
 
#【5】可直接调用accuracy_score计算准确率  
tra_label = classifier.predict(train_data)      #训练集的预测标签  
tes_label = classifier.predict(test_data)       #测试集的预测标签  
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

#【6】查看决策函数  
print('train_decision_function:\n', classifier.decision_function(train_data))     # (90,3)