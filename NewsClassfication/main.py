from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#数据的处理与分割
news = fetch_20newsgroups(subset = 'all')
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)

#使用朴素贝叶斯分类器对新闻文本数据进行预测
#文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#模型训练及预测
mnb = MultinomialNB()  #初始化贝叶斯模型
mnb.fit(X_train, y_train)  #模型训练
y_predict = mnb.predict(X_test)  #对测试样本进行预测

#模型评估
print('The accuracy of Navie Bayes Classifier is', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names = news.target_names))
