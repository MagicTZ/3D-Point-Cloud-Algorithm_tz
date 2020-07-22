# 文件功能： 实现 K-Means 算法

import numpy as np
import copy as cp
import random

def get_closest_dist(point, centroids):
    min_dist = np.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = np.sum((np.array(centroid) - np.array(point))**2) # 标准差
        if dist < min_dist:
            min_dist = dist
    return min_dist

# 功能：使用轮盘法初始化聚类中心
# 输入：
#       data: array, 输入数据
#       k：int，类别数量
# 输出：
#       cluster_centers: 聚类中心
def get_initial(data: np.array, k: int) -> list:
    cluster_centers = []
    data = data.tolist()
    cluster_centers.append(random.choice(data))
    d = [0 for _ in range(len(data))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data[i])
            break
    return cluster_centers

class K_Means(object):
    cluster_center = []
    n_points_ = 0

    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    # 功能：根据Dataset拟合模型
    # Input:
    #       data: 原始数据集, N*D（N: 样本数；D: 数据维度）
    def fit(self, data, method = 'mean'):
        # 作业1
        # 屏蔽开始
        if data is None:
            return False
        k = self.k_
        N, D = data.shape
        # 初始化
        data_center = np.zeros((k, D)) # 初始点 K*D
        new_data_center = np.zeros((k, D)) # K*D 
        # 随机在N个数据点中选取k个初始点
        #seed_idx = random.sample(list(range(N)),k)
        #data_center[:,:] = data[seed_idx,:]
        # 改进：使用Kmeans++选取初始点
        data_center = get_initial(data, k)

        #print('----------------Initial Points---------------')
        #print(data_center)   

        loss = 1000
        itr = 0
        tolerance =np.inf
        while tolerance > self.tolerance_ and itr < self.max_iter_:
            # Expectation Step (E step): fix expectation
            label_idx = np.zeros((N,1),dtype = int) # N * 1
            for i in range(N): # 每一个点找到最近的中心（可用kdtree和octree改进）
                dis_min = np.inf
                label = 0 # 标签
                for j in range(k):
                    temp = np.sum((data[i] - data_center[j])**2) # 使用标准差来判断距离
                    if temp < dis_min:
                        dis_min = temp
                        label = j   # 修改该点标签
                label_idx[i,:] = label

            new_data = np.hstack((label_idx, data))   

            # Maximum Step (M step)
            # 计算新的中心点位置
            # 方法1：计算mean作为聚类中心
            if method == 'mean':
                for class_n in range(k):
                    c_n = 0
                    for i in range(N):
                        if new_data[i,0] == class_n:
                            new_data_center[class_n,:] += new_data[i, 1:k+1] # K*D
                            c_n += 1
                    new_data_center[class_n, :] = new_data_center[class_n, :] / c_n # 新中心点
            # 方法2：计算medoid作为聚类中心
            #if method == 'medoid':
            
            #print(new_data_center)
            
            # 更新loss 
            old_loss = loss
            loss =np.sum(np.linalg.norm(new_data_center - data_center, axis = 1))

            data_center = cp.deepcopy(new_data_center)
            tolerance = abs(loss-old_loss)
            itr += 1
            
        self.cluster_center = cp.deepcopy(data_center)
        # 屏蔽结束

    # 功能：对输入的数据进行分类
    # Input:
    #       p_datas: 输入数据 N×D，D应该与类别大小一致
    # Output:
    #       result: 分类结果 N×1
    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        if p_datas is None:
            return False

        N =  p_datas.shape[0] # 样本数
        k = self.k_ # 类别数
        data_center = cp.deepcopy(self.cluster_center)

        #data_class_idx = np.zeros((N,1),dtype = int) # N * 1
        for i in range(N): # 每一个点找到最近的中心（可用kdtree和octree改进）
            dis_min = 100000000
            for j in range(k):
                dis = np.linalg.norm(p_datas[i,:] - data_center[j,:]) # 计算第i个点距离第j个中心点的距离
                if dis < dis_min: # 得到离中心
                    dis_min = dis # 更新最短距离
                    label = j # 修改该点类别
                    #data_class_idx[i,:] = j
            result.append(label)
        #print(data_class_idx)
        #new_data = np.hstack((data_class_idx, p_datas)) 
        #print(new_data)   
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

