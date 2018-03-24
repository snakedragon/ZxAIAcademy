
import numpy as np
from time import sleep
import json
import urllib



"""
function: load data from txt file
input parameter：  filename of data
output:  dataFields, labelField, number of features, number of label field
"""
def load_dataset(filename):

    with open(filename) as f:
        firstLine = f.readline()
        numFeature = len(firstLine.strip().split('\t'))-1

    print('dataset feature number is:%d'%numFeature)


    dataSet=[]
    labelSet=[]

    with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    line_array = line.strip().split('\t')
                    fieldsx = line_array[0:-1]
                    fieldy = line_array[-1]

                    data_fields = [float(ii) for ii in fieldsx]
                    label_field = float(fieldy)

                    dataSet.append(data_fields)
                    labelSet.append(label_field)
                except:
                    continue



    return dataSet, labelSet,numFeature,1



def load_dataset_test():
    x, y, nf, nl = load_dataset('abalone.txt')
    xArray = np.array(x)
    yArray = np.array(y)
    print(xArray.shape)
    print(yArray.shape)
    print(nf)
    print(nl)



"""
standard regression solution, use matrix
input: xArray, yArray
ouput: ws (coefficiency of linear transform)

ordinary least square linear regression

"""
def standardRegression(xArray, yArray):

    xMat = np.mat(xArray); yMat=np.mat(yArray).T
    xTx = xMat.T * xMat

    if np.linalg.det(xTx)==0.0:
        raise ValueError("this matrix is singular, can not do inverse")
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

"""
#yArr and yHatArr both need to be arrays
error computation
"""
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()



def Numpy_Exercise():
    a = np.array([[-1,2],[2,3]])
    b = np.array([[3,4],[4,5]])
    print('\n a:\n',a)
    print('\n b:\n',b)
    ##转置
    print('\n a transpose:\n',a.T)
    ##共扼矩阵
    aMat = np.matrix(a)
    print('\n a H:\n',aMat.I)
    ##逆矩阵
    print('\n a inv:\n',np.linalg.inv(a)) # 求逆
    ##转置
    print('\n a transpose:\n',a.T)
    # a + b，矩阵相加
    print("\n a+b: \n",a+b)
    # a - b，矩阵相减
    print("\n a-b: \n",a-b)
    #2x2 矩阵，矩阵相乘
    print("\n a mul b:\n",np.multiply(a,b))
    #2x3矩阵，矩阵点乘
    print("\n a dot b: \n",a.dot(b))
    #2x3矩阵，矩阵点除
    print("\n a/b \n:",a/np.linalg.inv(b))
    #求迹
    print("\n a trace",np.trace(a))
    #特征，特征向量
    eigval,eigvec = np.linalg.eig(a)
    #eigval = np.linalg.eigvals(a)) #直接求解特征值
    print("\n a eig value:\n",eigval)
    print('\n a eig vector:\n',eigvec)


"""

"""

def gradientDescend(xArray, yArray, numIter = 150,alpha=0.001):

    xMat = np.matrix(xArray)
    yMat = np.matrix(yArray)

    rows,features = xArray.shape
    weights = np.ones(features, np.float64)

    for ii in range(numIter):
        yHat = xMat * weights
        deltws = - xMat.T * (yMat - yHat)
        weights = weights - alpha * deltws  # 权值的更新公式

    return weights

"""
"""
def stochGradientDescend(xArray, yArray,alpha=0.001):

    xMat = np.matrix(xArray)
    yMat = np.matrix(yArray)

    rows,features = xArray.shape
    weights = np.ones(features, np.float64)

    for row in range(rows):
        xi = xMat[row]
        yi = yMat[row]
        delta = -xi * (yi - xi * weights)
        weights = weights - alpha*delta

    return weights


def stochGradientDescendExt(xArray, yArray, numIter=150, alpha=0.001, adaptive=False):

    xMat = np.matrix(xArray)
    yMat = np.matrix(yArray)

    rows,features = xArray.shape
    weights = np.ones(features, np.float64)

    for iter in range(numIter):
        for row in range(rows):
            if (adaptive):
                alpha = 2 / (1.0 + iter + row) + 0.01
            xi = xMat[row]
            yi = yMat[row]
            delta = -xi * (yi - xi * weights)
            weights = weights - alpha*delta

    return weights


'''
localize weights linear regression
'''
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = np.zeros(np.shape(yArr))       #easier for plotting
    xCopy = np.mat(xArr)
    xCopy.sort(0)
    for i in range(np.shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print
        "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = np.mean(xMat, 0)  # calc mean then subtract it off
    xVar = np.var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T

    return wMat


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    # returnMat = zeros((numIt,n)) #testing code remove
    ws = np.zeros((n, 1));
    wsTest = ws.copy();
    wsMax = ws.copy()
    for i in range(numIt):
        print
        ws.T
        lowestError = np.inf;
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        # returnMat[i,:]=ws.T
    # return returnMat



def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, setNum)
    pg = urllib.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = np.zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = [];
        trainY = []
        testX = [];
        testY = []
        np.random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = np.mat(testX);
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
            # print errorMat[i,k]
    meanErrors = np.mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0);
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))