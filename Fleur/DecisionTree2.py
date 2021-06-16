# 文件功能：决策树分类鸢尾花数据集
# 代码整体思路:
# 1 . 先处理数据，shuffle函数随机抽取80%样本做训练集。
# 2 . 特征值离散化
# 3 . 用信息熵来递归地构造树
# 4 . 用构造好的树来判断剩下20%的测试集，求算法做分类的正确率
 
from sklearn import datasets
import math
import numpy as np
 
# 【1】获取信息熵
def getInformationEntropy(arr, leng):
    return -(arr[0] / leng * math.log(arr[0] / leng if arr[0] > 0 else 1) + arr[1] / leng * math.log(
        arr[1] / leng if arr[1] > 0 else 1) + arr[2] / leng * math.log(arr[2] / leng if arr[2] > 0 else 1))
 
# 【2】离散化特征一的值
def discretization(index):
    feature1 = np.array([iris.data[:, index], iris.target]).T
    feature1 = feature1[feature1[:, 0].argsort()]
 
    counter1 = np.array([0, 0, 0])
    counter2 = np.array([0, 0, 0])
 
    resEntropy = 100000
    for i in range(len(feature1[:, 0])):
        counter1[int(feature1[i, 1])] = counter1[int(feature1[i, 1])] + 1
        counter2 = np.copy(counter1)
        for j in range(i + 1, len(feature1[:, 0])):
            counter2[int(feature1[j, 1])] = counter2[int(feature1[j, 1])] + 1
            # print(i,j,counter1,counter2)
            # 贪心算法求最优的切割点
            if i != j and j != len(feature1[:, 0]) - 1:
                sum = (i + 1) * getInformationEntropy(counter1, i + 1) + (j - i) * getInformationEntropy(
                    counter2 - counter1, j - i) + (length - j - 1) * getInformationEntropy(np.array(num) - counter2, length - j - 1)
                if sum < resEntropy:
                    resEntropy = sum
                    res = np.array([i, j])
    res_value = [feature1[res[0], 0], feature1[res[1], 0]]
    print(res, resEntropy, res_value)
    return res_value
 
# 【3】计算合适的分割值
def getRazors():
    a = []
    for i in range(len(iris.feature_names)):
        print(i)
        a.append(discretization(i))
    return np.array(a)
 
# 【4】随机抽取80%的训练集和20%的测试集
def divideData():
    completeData = np.c_[iris.data, iris.target.T]
    np.random.shuffle(completeData)
    trainData = completeData[range(int(length * 0.8)), :]
    testData = completeData[range(int(length * 0.8), length), :]
    return [trainData, testData]
 
# 【5】
def getEntropy(counter):
    res = 0
    denominator = np.sum(counter)
    if denominator == 0:
        return 0
    for value in counter:
        if value == 0:
            continue
        res += value / denominator * math.log(value / denominator if value > 0 and denominator > 0 else 1)
    return -res
# 【6】寻找最大索引
def findMaxIndex(dataSet):
    maxIndex = 0
    maxValue = -1
    for index, value in enumerate(dataSet):
        if value > maxValue:
            maxIndex = index
            maxValue = value
    return maxIndex
 
# 【7】递归
def recursion(featureSet, dataSet, counterSet):
    if (counterSet[0] == 0 and counterSet[1] == 0 and counterSet[2] != 0):
        return iris.target_names[2]
    if (counterSet[0] != 0 and counterSet[1] == 0 and counterSet[2] == 0):
        return iris.target_names[0]
    if (counterSet[0] == 0 and counterSet[1] != 0 and counterSet[2] == 0):
        return iris.target_names[1]
    if len(featureSet) == 0:
        return iris.target_names[findMaxIndex(counterSet)]
    if len(dataSet) == 0:
        return []
    res = 1000
    final = 0
    # print("剩余特征数目", len(featureSet))
    for feature in featureSet:
        i = razors[feature][0]
        j = razors[feature][1]
        # print("i = ",i," j = ",j)
        set1 = []
        set2 = []
        set3 = []
        counter1 = [0, 0, 0]
        counter2 = [0, 0, 0]
        counter3 = [0, 0, 0]
        for data in dataSet:
            index = int(data[-1])
            # print("data ",data," index ",index)
 
            if data[feature] < i:
                set1.append(data)
                counter1[index] = counter1[index] + 1
            elif data[feature] >= i and data[feature] <= j:
                set2.append(data)
                counter2[index] = counter2[index] + 1
            else:
                set3.append(data)
                counter3[index] = counter3[index] + 1
 
        a = (len(set1) * getEntropy(counter1) + len(set2) * getEntropy(counter2) + len(set3) * getEntropy(
            counter3)) / len(dataSet)
        # print("特征编号：",feature,"选取该特征得到的信息熵:",a)
        if a < res:
            res = a
            final = feature
    # 返回被选中的特征的下标
    # sequence.append(final)
    # print("最终在本节点上选取的特征编号是:",final)
    featureSet.remove(final)
    child = [0, 0, 0, 0]
    child[0] = final
    child[1] = recursion(featureSet, set1, counter1)
    child[2] = recursion(featureSet, set2, counter2)
    child[3] = recursion(featureSet, set3, counter3)
    return child
 
# 【8】决策
def judge(data, tree):
    root = "unknow"
    while (len(tree) > 0):
        if isinstance(tree, str) and tree in iris.target_names:
            return tree
        root = tree[0]
        if (isinstance(root, str)):
            return root
        if isinstance(root, int):
            if data[root] < razors[root][0] and tree[1] != []:
                tree = tree[1]
            elif tree[2] != [] and (tree[1] == [] or (data[root] >= razors[root][0] and data[root] <= razors[root][1])):
                tree = tree[2]
            else:
                tree = tree[3]
    return root
 
# 【9】调用
if __name__ == '__main__':
    iris = datasets.load_iris()
    num = [0, 0, 0]
    for row in iris.data:
        num[int(row[-1])] = num[int(row[-1])] + 1
    length = len(iris.target)
    [trainData, testData] = divideData()
    razors = getRazors()
    tree = recursion(list(range(len(iris.feature_names))), trainData,
                     [np.sum(trainData[:, -1] == 0), np.sum(trainData[:, -1] == 1), np.sum(trainData[:, -1] == 2)])
    print("本次选取的训练集构建出的树： ", tree)
    index = 0
    right = 0
    for data in testData:
        result = judge(testData[index], tree)
        truth = iris.target_names[int(testData[index][-1])]
        print("result is ", result, "  truth is ", truth)
        index = index + 1
        if result == truth:
            right = right + 1
    print("正确率 ： ", right / index)