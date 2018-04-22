from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) + 1
    dataArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        curLine.insert(0, '1')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
    dataArr = array(dataArr)
    return dataArr


def pLearn(eta=0.02):
    dataArr = loadDataSet('exdata.txt')
    ws = initPara2(dataArr)
    dataMat = mat(dataArr)
    m, n = shape(dataMat)
    for j in range(200):
        for i in range(m):
            label = dataMat[i, -1]
            xMat = mat(dataMat[i, 0:-1])
            if label * ws * xMat.T < 0:
                ws = ws + eta * label * xMat
    psamplenum, tempn = shape(mat(dataArr[dataArr[:, -1] == 1, :]))
    perror = 0.0
    nsamplenum, temnn = shape(mat(dataArr[dataArr[:, -1] == -1, :]))
    nerror = 0.0
    for i in range(m):
        label = dataMat[i, -1]
        xMat = mat(dataMat[i, 0:-1])
        if label * ws * xMat.T < 0:  # count error
            if label == 1:
                perror = perror + 1  # error of positive
            else:
                nerror = nerror + 1  # error of negative
    return dataArr, ws, psamplenum, perror, nsamplenum, nerror


def initPara1(dataArr):
    pSample = dataArr[dataArr[:, -1] == 1, :]
    nSample = dataArr[dataArr[:, -1] == -1, :]
    ws = mat(sum(pSample, axis=0) - sum(nSample, axis=0))
    m, n = shape(ws)
    ws = ws[:, 0:n - 1]
    return ws


def initPara2(dataArr):
    m, n = shape(mat(dataArr))
    ws = mat(ones(n - 1))
    return ws


def drawResult(dataArr, ws, ps, pe, ns, ne):
    pSample = dataArr[dataArr[:, -1] == 1, :]
    nSample = dataArr[dataArr[:, -1] == -1, :]
    fig = plt.figure(None, None, None, 'w')
    ax = fig.add_subplot(111)
    ax.scatter(pSample[:, 1], pSample[:, 2], c='r', marker='o')
    ax.scatter(nSample[:, 1], nSample[:, 2], c='b', marker='*')
    xmin = min(dataArr[:, 1])
    xmax = max(dataArr[:, 1])
    xCopy = arange(xmin - 5, xmax + 5, 0.5)
    yHat = -xCopy * ws[0, 1] / ws[0, 2] - ws[0, 0] / ws[0, 2]
    ax.plot(xCopy, yHat, 'k')
    s1 = 'recall of positive:' + str((1 - pe / ps) * 100) + '%'
    s2 = 'recall of negative:' + str((1 - ne / ns) * 100) + '%'
    ax.text(-7.9, 15, s1)
    ax.text(-7.9, 14, s2)
    plt.show()


if __name__ == '__main__':
    dArr, ws, ps, pe, ns, ne = pLearn()
    drawResult(dArr, ws, ps, pe, ns, ne)
