# 文件功能： 实现 K-Means 算法

import numpy as np
import copy as cp

class K_Means(object):
    cluster_center = []
    
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    # 功能：根据Dataset拟合模型
    # Input:
    #       data: 原始数据集, N*D（N: 样本数；D: 数据维度）
    def fit(self, data):
        # 作业1
        # 屏蔽开始
        # 初值选取 (从原始数据集中随机选点作为类别中心)
        if data is None:
            return False

        k = self.k_
        data_center = np.zeros((k, data.shape[1])) # 初始点 K*D
        new_data_center = np.zeros((k, data.shape[1])) # K*D
        N = data.shape[0]
        seed_idx = -1 # 用来存储初始点的索引
        loss = 1000
        itr = 0
        
        # 随机在N个数据点中选取k个初始点
        for i in range(k):
            seed_idx = np.random.randint(0,N)
            data_center[i,:] = data[seed_idx,:]
        
        print('----------------Initial Points---------------')
        print(data_center)   

        while loss > self.tolerance_ and itr < self.max_iter_:
            # Expectation Step (E step): fix expectation
            data_class_idx = np.zeros((N,1),dtype = int) # N * 1
            for i in range(N): # 每一个点找到最近的中心（可用kdtree和octree改进）
                dis_min = 100000000
                for j in range(k):
                    temp = np.linalg.norm(data[i,:] - data_center[j,:])
                    if temp < dis_min:
                        dis_min = temp
                        # 修改该点类别
                        data_class_idx[i,:] = j
            #print(data_class_idx)
            new_data = np.hstack((data_class_idx, data)) 
            #print(new_data)      

            # Maximum Step (M step)
            # 计算新的中心点位置（这里直接用mean)
            for class_n in range(k):
                c_n = 0
                for i in range(N):
                    if new_data[i,0] == class_n:
                        new_data_center[class_n,:] += new_data[i, 1:k+1] # K*D
                        c_n += 1
                new_data_center[class_n, :] = new_data_center[class_n, :] / c_n # 新中心点
            #print(new_data_center)
            
            # 更新loss 
            old_loss = loss
            loss = np.linalg.norm(new_data_center - data_center)

            # 只有在loss减少的时候才更新中心点
            if loss < old_loss:
                data_center = cp.deepcopy(new_data_center)
            itr += 1
            
        print('loss: ',loss)
        print('itr: ',itr)
        print('分类中心:', data_center)
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
        N =  p_datas.shape[0]
        k = self.k_
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

