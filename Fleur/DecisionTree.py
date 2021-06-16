from sklearn import datasets                         # 导入方法类
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
 
# 【1】载入数据集
iris = datasets.load_iris()                         # 加载 iris 数据集
iris_feature = iris.data                            # 特征数据
iris_target = iris.target                           # 分类数据
 
# 【2】数据集划分
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33, random_state=42)
 

# 【3】训练模型
dt_model = DecisionTreeClassifier()                 # 所有参数均置为默认状态
dt_model.fit(feature_train,target_train)            # 使用训练集训练模型
predict_results = dt_model.predict(feature_test)    # 使用模型对测试集进行预测
  
# 【4】结果评估
scores = dt_model.score(feature_test, target_test)
print(scores)