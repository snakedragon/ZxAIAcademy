
#-*- encoding=utf-8 -*-


import numpy as np
import math

#load dataset
names = ['sepal_length','sepal_width','petal_length','petal_width','class']


def load_data(filename):

    features=[]
    labels=[]

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            lineArray=line.strip().split(',')
            feature = lineArray[0:-1]
            feature = [float(i) for i in feature]
            features.append(feature)
            labels.append(lineArray[-1])

    return np.array(features),np.array(labels)


"""
knn算法
"""
def classift(inX, dataSet, labels, k):

    # shape获取array的大小
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet		# 矩阵减法
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 行向量求和
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
		# 排序
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]


def SIRIClassify(filename,k=3,ratio=0.3):
    hoRatio = ratio
    features, labels = load_data(filename)
    m = features.shape[0]
    numTestVecs = int(m*hoRatio)
    tureCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classift(features[i, :], features[numTestVecs:m, :], labels[numTestVecs:m], k)
        #print(classifierResult, labels[i])
        if classifierResult == labels[i]:
            tureCount += 1.0
    print("the total ture rate is: %f " % (tureCount/float(numTestVecs)))
    print("k is: %d " % k)
    print("ratio is: %d " % ratio)
    print(tureCount,numTestVecs)
    return features



for k in range(1,9,2):
    for ratio in range(1,4,1):
        SIRIClassify('iris.data.txt',k,ratio*0.1)




