import numpy as np
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt


def distance(x1, x2):
    """
    获得两个样本点之间的距离
    :param x1: 样本点1
    :param x2: 样本点2
    :return:
    """
    dist = np.sqrt(np.power(x1-x2,2).sum())
    return dist

def get_dist_matrix(data):
    """
    获取距离矩阵
    :param data: 样本集合
    :return: 距离矩阵
    """
    n = len(data)  #样本总数
    dist_matrix = np.zeros((n, n)) # 初始化邻接矩阵为n×n的全0矩阵
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = dist_matrix[j][i] = distance(data[i], data[j])
    return dist_matrix

class SC(object):

    def __init__(self, n_clusters, knn_k):
        self.n_clusters_  = n_clusters
        self.knn_k_ = knn_k

    def getW(self, data, k):
        """
        获的邻接矩阵 W
        :param data: 样本集合
        :param k : KNN参数
        :return: W
        """
        n = len(data)
        dist_matrix = get_dist_matrix(data)
        W = np.zeros((n, n))
        for idx, item in enumerate(dist_matrix):
            idx_array = np.argsort(item)  # 每一行距离列表进行排序,得到对应的索引列表
            W[idx][idx_array[1:k+1]] = 1
        transpW =np.transpose(W)
        return (W+transpW)/2

    def getD(self, W):
        """
        获得度矩阵
        :param W: 邻接矩阵
        :return: D
        """
        D = np.diag(sum(W))
        return D


    def getL(self, D,W):
        """
        获得拉普拉斯矩阵
        :param W: 邻接矩阵
        :param D: 度矩阵
        :return: L
        """
        return D-W

    def getEigen(self, L, cluster_num):
        """
        获得拉普拉斯矩阵的特征矩阵
        :param L:
        :param cluter_num: 聚类数目
        :return:
        """
        eigvec, eigval, _ = np.linalg.svd(L)
        ix = np.argsort(eigval)[0:cluster_num]
        return eigvec[:, ix]


    def fit(self, data):
        k = self.knn_k_
        cluster_num = self.n_clusters_
        data = np.array(data)
        W = self.getW(data, k)
        D = self.getD(W)
        L = self.getL(D,W)
        eigvec = self.getEigen(L, cluster_num)
        self.eigvec_ = eigvec
    
    def predict(self, data):
        clf = KMeans(n_clusters=self.n_clusters_)
        s = clf.fit(self.eigvec_)  # 聚类
        result = s.labels_
        return  result


if __name__ == '__main__':
    cluster_num = 3
    knn_k = 5
    data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    data = data[0:-1]  # 最后一列为标签列
    spectral_clustering = SC(n_clusters= 3, knn_k = 5)
    spectral_clustering.fit(data)
    label = spectral_clustering.predict(data)
    print(label)