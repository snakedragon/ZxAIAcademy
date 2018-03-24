import regression
import regressiongdc
from numpy import *
import matplotlib.pyplot as plt
def drawresult(filename):
	xArr,yArr=regression.loadDataSet(filename)
	ws=regression.standRegres(xArr,yArr)
	ws1=regressiongdc.gradDscent(xArr,yArr)
	xMat=mat(xArr)
	yMat=mat(yArr)
	fig=plt.figure(None,None,None,'w')
	ax=fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*ws
	ax.plot(xCopy[:,1],yHat,'k')
	yHat1=xCopy*ws1
	ax.plot(xCopy[:,1],yHat1,'*r')
	plt.show()

def drawsimpleresult(filename):
	xArr,yArr=regression.loadDataSet1(filename)
	ws=regression.standRegres(xArr,yArr)
	xMat=mat(xArr)
	yMat=mat(yArr)
	fig=plt.figure(None,None,None,'w')
	ax=fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*ws
	ax.plot(xCopy[:,1],yHat,'k')
	plt.show()

def drawstocGdc(filename):
	xArr,yArr=regressiongdc.loadDataSet(filename)
	ws=regressiongdc.stocGradDscent(xArr,yArr)
	ws1=regressiongdc.gradDscent(xArr,yArr)
	xMat=mat(xArr)
	yMat=mat(yArr)
	fig=plt.figure(None,None,None,'w')
	ax=fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*mat(ws).T
	yHat1=xCopy*ws1
	print ws
	print ws1
	ax.plot(xCopy[:,1],yHat,'r')
	ax.plot(xCopy[:,1],yHat1,'k')
	plt.show()

def drawstocGdc1(filename,numIter):
	xArr,yArr=regressiongdc.loadDataSet(filename)
	ws=regressiongdc.stocGradDscent1(xArr,yArr,numIter)
	ws1=regressiongdc.gradDscent(xArr,yArr)
	xMat=mat(xArr)
	yMat=mat(yArr)
	fig=plt.figure(None,None,None,'w')
	ax=fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*mat(ws).T
	yHat1=xCopy*ws1
	print ws
	print ws1
	ax.plot(xCopy[:,1],yHat,'r')
	ax.plot(xCopy[:,1],yHat1,'k')
	plt.show()

def drawstocGdc2(filename,numIter):
	xArr,yArr=regressiongdc.loadDataSet(filename)
	ws=regressiongdc.gradDscent(xArr,yArr)
	ws1=regressiongdc.stocGradDscent1(xArr,yArr,numIter)
	ws2=regressiongdc.stocGradDscent2(xArr,yArr,numIter)
	xMat=mat(xArr)
	yMat=mat(yArr)
	fig=plt.figure(None,None,None,'w')
	ax=fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
        yHat=xCopy*ws
	yHat1=xCopy*mat(ws1).T
	yHat2=xCopy*mat(ws2).T
        print ws
	print ws1
	print ws2
	ax.plot(xCopy[:,1],yHat,'b')
	ax.plot(xCopy[:,1],yHat1,'r')
	ax.plot(xCopy[:,1],yHat2,'k')
	plt.show()
