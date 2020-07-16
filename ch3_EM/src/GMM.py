# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

import KMeans as km

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    # 屏蔽开始
    # 更新W
    

    # 更新pi
 
        
    # 更新Mu


    # 更新Var


    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始

        return False
        # 屏蔽结束
    
    def predict(self, data):
        result = []
        # 屏蔽开始

        # 屏蔽结束
        return result


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1) # 产生高斯分布的多维变量数据
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

# 功能: 显示分类结果，从而直观地检查分类是否正确
# Input：
#       label: 利用预测模型得到的标签
#       X:  原始数据
def clusterShow(label, X):
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    C1 =[]
    C2 = []
    C3 =[]
    for x, i in zip(X, range(X.shape[0])):
        if label[i] == 0:
            C1.append(x)
        if label[i] == 1:
            C2.append(x)
        if label[i] == 2:
            C3.append(x)
    k1 = np.array(C1)
    k2 = np.array(C2)
    k3 = np.array(C3)
    plt.scatter(k1[:, 0], k1[:, 1], s=5)
    plt.scatter(k2[:, 0], k2[:, 1], s=5)
    plt.scatter(k3[:, 0], k3[:, 1], s=5)    
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    # K-means
    kmeans = km.K_Means(n_clusters=3)
    kmeans.fit(X)
    cat = kmeans.predict(X)
    print(cat)
    
    clusterShow(cat, X) # 显示预测结果

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

