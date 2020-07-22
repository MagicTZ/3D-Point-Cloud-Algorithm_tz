# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math
import time 

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

import KMeans as km
import Spectralclustering as sc

class GMM(object):
    pi_  = []
    w_ = []
    Mu_ = []
    Var_ = []
    Nk_ =[]

    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
    
    # 屏蔽开始
    # 计算对数
    def get_log(self, X, Pi, Mu, Var):
        N, K = len(X), len(Pi)
        p = np.zeros((N,K))
        for label in range(K):
            p[:, label] = Pi[label] * multivariate_normal.pdf(X, Mu[label], np.diag(Var[label]))
        return np.mean(np.log(p.sum(axis=1)))

    def plot_clusters(self, X, Mu, Var, Mu_true=None, Var_true=None):
        colors = ['b', 'g', 'r']
        n_clusters = len(Mu)
        plt.figure(figsize=(10, 8))
        plt.axis([-10, 15, -5, 15])
        plt.scatter(X[:, 0], X[:, 1], s=5)
        ax = plt.gca()
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
            ellipse = Ellipse(Mu[i], 3 * Var[i][0][0], 3 * Var[i][1][1], **plot_args)
            ax.add_patch(ellipse)
        if (Mu_true is not None) & (Var_true is not None):
            for i in range(n_clusters):
                plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
                ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
                ax.add_patch(ellipse)         
        plt.show()

    # 更新W（后验）: 每个样本属于每一个cluster的概率
    # 输出：
    #       W：posterior, N*K
    def update_w(self, X: np.array, Mu, Var, Pi)->np.array:
        # 样本数和特征数
        N = len(X)
        K = len(Pi)
        p = np.zeros((N, K)) # pdf N*K
        for i in range(K):
            var = Var[i,:,:]
            p[:, i] = multivariate_normal.pdf(X, Mu[i,:], var, allow_singular= True) * Pi[i]
        W_sum =np.sum(p,axis =1).reshape(-1,1) # n_points*1
        W = p/W_sum # n_points*n_clusters
        return W

    # 更新pi：求出Nk和每一个cluster的比重 Pi = Nk/N
    # 输入：
    #       W: 通过Estep求得的权重矩阵
    # 输出：
    #       Pi: 1×K，每一个cluster的比重
    def update_pi(self, W):
        self.Nk_ = np.sum(W, axis = 0) # 1*K 每个类别分别有多少点
        Pi = self.Nk_ / np.sum(W)
        return Pi.reshape(-1,1)

    # 更新Mu
    # 输出：
    #       Mu: 聚类中心，K*D
    def update_mu(self, X, W):
        n_clusters = self.n_clusters_

        Mu = np.zeros((n_clusters, X.shape[1])) # K*D
        for i in range(n_clusters):
            #Mu[i, :] = np.average(X, axis =0, weights=W[:,i])
            Mu[i, :] = np.dot(W[:, i].T, X)
            #print(Mu[i,:])
        Mu = Mu / self.Nk_.reshape(-1,1)
        return Mu 

    # 更新Var
    # 输出：
    #       Var：协方差矩阵，K×D×D
    def update_var(self, X, Mu, W):
        D = X.shape[1]
        Var = np.zeros((self.n_clusters_, D, D)) # Var: K*D*D
        for i in range(self.n_clusters_):
            deviation = X - Mu[i,:] # N*D
            A = np.diag(W[:,i])
            Var[i, :, :] = np.dot(deviation.T, np.dot(A, deviation)) / self.Nk_[i]# var = UT*A*U
        return Var
 
    
    # 屏蔽结束

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # Initialization
        n_clusters = self.n_clusters_
        n_points  = len(data)
        D = data.shape[1]
        # 随机在N个数据点中选取k个初始点
        # Kmeans 轮盘法选初始点
        #Mu = km.get_initial(data, n_clusters)  # 聚类中心
        #seed_idx = random.sample(list(range(n_points)),n_clusters)
        #for seed in seed_idx:
        #    Mu.append(data[seed,:])
        kmean = km.K_Means(n_clusters= n_clusters, max_iter=30)
        kmean.fit(data)
        Mu = kmean.cluster_center
        Var = np.asarray([np.cov(data, rowvar = False)]*n_clusters) # 方差: K*D*D
        #Var = np.ones((n_clusters, D, D))
        pi = [1/n_clusters]*n_clusters # 每一个cluster的比重： pi =[1/k, 1/k, 1/k]
        w = np.ones((n_points, n_clusters))/n_clusters  # 每一个变量分类权重
        #pi = w.sum(axis = 0) / w.sum()  
        #迭代求解
        log_p = 1
        old_log_p =0
        loglh = []
        time_w, time_pi, time_mu, time_var = 0 ,0 ,0 ,0
        for i in range(self.max_iter_):
            #self.plot_clusters(X, Mu, Var)
            old_log_p = log_p
            # E step
            # Update weight (posterior)
            time_start = time.time()
            w = self.update_w(data, Mu, Var, pi) # 
            time_w += time.time() -time_start
            # M step
            # 更新pi
            time_start = time.time()
            pi = self.update_pi(w)
            time_pi += time.time() -time_start
            # 更新聚类中心
            time_start = time.time()
            Mu = self.update_mu(data, w)
            time_mu += time.time() -time_start
            # 更新协方差矩阵
            time_start = time.time()
            Var = self.update_var(data, Mu, w)
            time_var += time.time() -time_start

            log_p = self.get_log(data, pi, Mu, Var)
            #loglh.append(log_p)
            #print('log-likehood:%.3f'%loglh[-1])
            if abs(log_p-old_log_p) < 0.001:
                #print(i)
                break

        # update parameters
        self.w_ = w
        self.pi_ = pi
        self.Mu_ = Mu
        self.Var_ = Var
        print("时间：", time_w, time_pi, time_mu, time_var)
        # 屏蔽结束
    
    def predict(self, data):
        result = []
        # 屏蔽开始
        W = self.update_w(data, self.Mu_, self.Var_, self.pi_)

        result = np.argmax(W, axis = 1)
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
    
    #show_cluster(cat, X) # 显示预测结果
    
    # GMM
    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    show_cluster(cat, X)
    
    spectral_clustering = sc.SC(n_clusters = 3, knn_k = 5)
    spectral_clustering.fit(X)
    label = spectral_clustering.predict(X)
    print(label)
    show_cluster(label,X)
    # 初始化

    

