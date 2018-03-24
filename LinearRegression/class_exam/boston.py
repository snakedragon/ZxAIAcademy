from numpy import *


# 导入数据波斯特房价数据集，第四列为标签
def loadDataSet(fileName):

    numFeat = len(open(fileName).readline().split('\t')) - 1

    dataMat = []
    labelMat = []

    fr = open(fileName)

    j=0

    for line in fr.readlines():
        j=j+1
        lineArr = []

        curLine = [z.strip() for z in line.strip().split('\t') if len(z.strip()) > 0]

        if len(curLine)<numFeat+1:
            curLine = [z.strip() for z in line.strip().split(' ') if len(z.strip()) > 0]
            if len(curLine)<numFeat+1:
                continue
        else:
            pass


        try:
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))

            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        except:
            print(i)
            pass

    return dataMat, labelMat


# 求解500次梯度下降后的更新权值
def gradDscent(xArr, yArr,maxCycles=10000):
    xMat = mat(xArr)
    yMat = mat(yArr).T  # 得到标签矩阵的转置
    m, n = shape(xMat)  # 得到矩阵的大小，行与列
    alpha = 0.001   # 学习率
    # 迭代次数
    weights = ones((n, 1))   # 初始化权值
    for k in range(maxCycles):
        yHat = xMat * weights
        deltws = xMat.T * (yMat - yHat)
        weights = weights + alpha * deltws  # 权值的更新公式
    return weights


# 求解样本数随机梯度下降的更新后的权值
def stocGradDscent(xArr, yArr):
    xMat = array(xArr)
    m, n = shape(xMat)
    alpha = 0.001
    weights = ones(n)
    for i in range(m):
        yHat = sum(xMat[i] * weights)
        deltws = (yArr[i] - yHat) * xMat[i]
        weights = weights + alpha * deltws
    return weights


# 求解150次随机梯度下降的更新后的权值
def stocGradDscent1(xArr, yArr, numIter=1000):
    xMat = array(xArr)
    m, n = shape(xMat)
    alpha = 0.001
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            yHat = sum(xMat[i] * weights)
            deltws = (yArr[i] - yHat) * xMat[i]
            weights = weights + alpha * deltws
    return weights


# 可以根据迭代次数，调整学习率，求解150次随机梯度下降的更新后的权值
def stocGradDscent2(xArr, yArr, numIter=10000):
    xMat = array(xArr)
    m, n = shape(xMat)
    weights = ones(n)
    dataIndex = range(m)
    for j in range(numIter):
        for i in dataIndex:
            alpha = 2 / (1.0 + j + i) + 0.01
            # randIndex=int(random.uniform(0,len(dataIndex)))
            # yHat=sum(xMat[randIndex]*weights)
            yHat = sum(xMat[i] * weights)
            # deltws=(yArr[randIndex]-yHat)*xMat[randIndex]
            deltws = (yArr[i] - yHat) * xMat[i]
            weights = weights + alpha * deltws
    return weights


def ridgeRegres(xMat, yMat, lam=0.2):

    xMat = matrix(xMat)
    yMat = matrix(yMat)

    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    A= (xMat.T * yMat)
    ws = denom.I * A
    return ws


def ridgeTest(xArr, yArr):

    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

# 数据归一化
def autoNorm(dataSet):

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals

def autoNorm1(dataSet):

    datax = array(dataSet)
    minValue = min(datax)
    maxValue = max(datax)
    ranges = maxValue-minValue
    normalDataset = ranges/(dataSet-minValue)
    return normalDataset






from sklearn import preprocessing

data, label = loadDataSet('boston.txt')
print(label)
print(shape(data))

scaler = preprocessing.MinMaxScaler()
zdata = scaler.fit_transform(data)
print(zdata)

"""

print("-------------随机梯度下降2----------------")

for numIter in range(1000,21000,3000):
    weights2 = stocGradDscent2(zdata,label,numIter)
    print("number of iteration is:" + str(numIter))
    print(weights2)

print("------------随机梯度下降1-----------------")

for numIter in range(1000,21000,3000):
    weights1 = stocGradDscent1(zdata,label,numIter)
    print("number of iteration is:" + str(numIter))
    print(weights1)

print("-----------------------------")

"""
for numIter in range(10000,100000,10000):
    weights = gradDscent(zdata,label,numIter)
    print(weights)

#weights3 = stocGradDscent(zdata,label)
#print(weights3)

#print("-----------------------------")
#weights4= gradDscent(zdata,label)
#print(weights4)

#weights5= ridgeRegres(zdata,label)
#print(weights5)
