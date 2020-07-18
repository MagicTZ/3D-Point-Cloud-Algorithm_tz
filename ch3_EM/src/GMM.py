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
    pi_  = []
    w_ = []
    Mu_ = []
    Var_ = []

    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
    
    # 屏蔽开始
    # 更新W: 每个样本属于每一个cluster的概率
    def update_w(self, X, Mu, Var, Pi):
        n_points = len(X)
        n_clusters = len(Pi)
        p = np.zeros((n_points, n_clusters)) # pdf
        for i in range(n_clusters):
            var = np.diag(Var[i])
            p[:, i] = multivariate_normal.pdf(X, Mu[i], var) * Pi[i]
        W_sum =np.sum(p,axis =1).reshape(-1,1) # n_points*1
        W = p/W_sum # n_points*n_clusters
        return W

    # 更新pi：每一个cluster的比重 Pi = W^-1
    def update_pi(self, W):
        Pi = np.sum(W, axis = 0) / np.sum(W)
        return Pi

    # 更新Mu
    def update_mu(self, X, W):
        n_clusters = self.n_clusters_

        Mu = np.zeros((n_clusters, X.shape[1])) # K*D
        for i in range(n_clusters):
            Mu[i] = np.average(X, axis =0, weights=W[:,i])
        #Mu = np.sum(Mu, axis = 0) / np.sum(W, axis=0)
        return Mu 

    # 更新Var
    def update_var(self, X, Mu, W):
        Var = np.zeros((self.n_clusters_, X.shape[1]))
        for i in range(self.n_clusters_):
            Var[i] = np.average((X - Mu[i])**2, axis = 0, weights = W[:,i])

        return Var
 
    
    # 屏蔽结束
    def __get_expectation(self, data):
        """
        Update posteriori
        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray
        """
        N,_ = data.shape
        posteriori = np.zeros((self.n_clusters_, N))  
        priori = np.ones((self.n_clusters_, 1)) / self.n_clusters_

        # expectation:
        for k in range(self.n_clusters_):
            posteriori[k] = multivariate_normal.pdf(
                data, 
                mean=self.Mu_[k], cov=self.Var_[k]
            )
        # get posteriori:
        posteriori = np.dot(
            np.diag(priori.ravel()), posteriori
        )
        # normalize:
        posteriori /= np.sum(posteriori, axis=0) 
        return posteriori  

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # Initialization
        n_clusters = self.n_clusters_
        n_points  = len(data)
        Mu = [[0,0],[2, 3],[6,1]] # 改进方法：随机选取三个点
        Var = [[1,1],[1,1],[1,1]] 
        pi = [1/n_clusters]*n_clusters # 每一个cluster的比重： pi =[1/k, 1/k, 1/k]
        w = np.ones((n_points, n_clusters))/n_clusters
        pi = w.sum(axis = 0) / w.sum()  
        #迭代求解
        for _ in range(self.max_iter_):
            # E step
            w = self.update_w(data, Mu, Var, pi) # 
            pi = self.update_pi(w)
            # M step
            Mu = self.update_mu(data, w)
            Var = self.update_var(data, Mu, w)

        # update parameters
        self.w_ = w
        self.pi_ = pi
        self.Mu_ = Mu
        self.Var_ = Var
        # 屏蔽结束
    
    def predict(self, data):
        result = []
        # 屏蔽开始
        posteriori = self.__get_expectation(data)

        result = np.argmax(posteriori, axis = 0)
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
def show_cluster(label, X):
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
    
    show_cluster(cat, X) # 显示预测结果
    
    # GMM
    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    show_cluster(cat, X)
    # 初始化

    

