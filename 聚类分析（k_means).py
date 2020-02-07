from numpy import *
import numpy as np
import time

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltline = map(float, curline)
        dataMat.append(fltline)
    return dataMat


#计算两个向量的距离，欧式距离
def distE(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

#随机选择中心点
def randCent(data, k):
    dataSet = np.array(data)
    n = shape(dataSet)[1]
    centriods = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j])-minJ)
        centriods[:, j] = minJ+rangeJ*np.random.rand(k, 1)
    return centriods

def kmeans(data, k, disMea=distE, createCent=randCent):
    dataSet = np.array(data)
    m = shape(dataSet)[0]
    clusterA = mat(zeros((m, 2)))
    centriods = createCent(dataSet, k)
    clusterC =True
    while clusterC:
        clusterC = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = disMea(centriods[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterA[i, 0] != minIndex:
                clusterC = True

            clusterA[i, :] = minIndex, minDist**2
        print(centriods)

        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterA[:, 0].A == cent)[0]]  # get all the point in this cluster
            centriods[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centriods, clusterA

def show(dataSet, k, centriods, clusterA):
     import matplotlib.pyplot as plt
     dataSet = np.array(dataSet)
     numSamples, dim = len(dataSet), len(dataSet[0])
     mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
     for i in range(numSamples):
         markIndex = int(clusterA[i, 0])
         plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

     mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
     for i in range(k):
         plt.plot(centriods[i, 0], centriods[i, 1], mark[i], markersize=12)
     plt.show()

def main():
    data = []
    for i in range(200):
        data.append([np.random.uniform(0,100), np.random.uniform(0,100)])
    myCentroids, clustAssing = kmeans(data, 4)
    print(myCentroids)
    show(data, 4, myCentroids, clustAssing)

if __name__ == '__main__':
    main()
