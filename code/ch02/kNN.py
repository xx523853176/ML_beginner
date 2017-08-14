#! usr/bin/env python
# -*- coding:utf-8 -*-

'kNN'
__author__ = 'HUSKY'

import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
         #（基准点inX，dataSet数据库，labels数据标签，k取值个数）
    dataSetSize = dataSet.shape[0]     #dataSet的行数，即data个数
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
         #算差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
         #算距离
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
             #最小k个点
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
             #获取其中每个coteIlabel的个数，成dict
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]
    
def createDataSet():     #创建数据集、标签
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()
x = classify0([0,0], group, labels, 3)
print(x)
