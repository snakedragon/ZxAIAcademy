# coding=utf-8
from numpy import *
import operator


def classift(inX, dataSet, labels, k):
    """
        knn算法
    """
    # shape获取array的大小
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet		# 矩阵减法
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 行向量求和
    distances = sqDistances**0.5  
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
		# 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]


# 读取文件
def fileMatrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件的长度
    numberOfLines = len(fr.readlines())
    # 建立一个初始化的1000行4列的0矩阵
    returnMat = zeros((numberOfLines, 4))
    # 空列表
    classLabelVector = []
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[0:4]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

# 测试代码
def datingClassTest():
    hoRatio = 0.8
    datingDataMat, datingLabels = fileMatrix("iris_.data.txt")
    m = datingDataMat.shape[0]
    numTestVecs = int(m*hoRatio)
    tureCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classift(datingDataMat[i, :], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(classifierResult, datingLabels[i])
        if classifierResult == datingLabels[i]:
            tureCount += 1.0
    print("the total ture rate is: %f" % (tureCount/float(numTestVecs)))
    print(tureCount,numTestVecs)
    return datingDataMat

datingClassTest()

print(datingClassTest()[:, 1],'\neeeeeeeeeeeeeeeeeeee\n',datingClassTest()[:, 2])
#  测试机样本数量
# # 定义figure
# fig = plt.figure()
# # ax = fig.add_subplot(349),
# # 参数349的意思是：将画布分割成3行4列，图像画在从左到右从上到下的第9块
# ax = fig.add_subplot(111)
# # s=size大小 ，c = color颜色 ， marker标记的符号 ，alpha=（0~1）
# ax.scatter(datingClassTest()[:, 1], datingClassTest()[:, 2],  s=20, c='R', marker='', alpha=0.5, label='C1')
# plt.show()