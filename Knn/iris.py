'''
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
'''

import pandas as pd

#load dataset
names = ['sepal_length','sepal_width','petal_length','petal_width','class']

#in CSV fromat without a header line so we
df = pd.read_csv('iris.data.txt',header=None,names=names)
df.head()