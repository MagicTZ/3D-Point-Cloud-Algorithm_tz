# Author: MagicTZ
# Date: 2020.07.23
# Github: https://github.com/MagicTZ/3D-Point-Cloud-Algorithm_tz

# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import math
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

import open3d as o3d

import voxel_filter as filter # 采样方法
from sklearn.neighbors import KDTree # KDTree 进行搜索
import time # 计时


def read_velodyne_bin(path):
    '''从kitti的.bin格式点云文件中读取点云

    Args:
        path: 文件路径
    
    Returns:
        homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def estimate_normal(point, normalize = True):
    '''法向量估计

    已知3个平面点point, 求平面方程，返回系数矩阵

    Args:
        point: 3*3 array
        normalize: bool
    
    Returns:
        coef:  1*4 array
    '''
    # 检查是否共线
    
    # ax+by+cz+d = 0
    # 法向量计算： a  = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    #                             b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    #                             c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    #                             d = -a*x1 - b*y1 - c*z1
    v21 = point[1,:] - point[0,:]
    v31 = point[2,:] - point[0,:]
    a = (v21[1]*v31[2]) - (v31[1]*v21[2])
    b = (v21[2]*v31[0]) - (v31[2]*v21[0])
    c = (v21[0]*v31[1]) - (v31[0]*v21[1])
    if normalize:
        r = math.sqrt(a**2+b**2+c**2)
        a= a/r
        b = b/r
        c = c/r
    d = - a*point[0,0] - b*point[0,1] - c*point[0,2]

    return np.array([a,b,c,d])
    
def ground_segmentation(data, sample = 3, max_iter = 100, dis_t = 0.35, e = 0.75):
    '''从点云文件中滤除地面点

    Args:
        data: 一帧完整点云, N*3
        sample: 随机选取的样本点数
        max_iter: 最大迭代次数
        dis_t: 距离阈值，判断是否属于inlier
        e: 非地面点占所有点的概率

    Returns:
        non_ground_point_idx: 非地面点索引
        ground_point_idx: 地面点索引
    '''
    N, D = data.shape # N：总点数， D：维度
    T = 0 # 地面点数量
    max_point = 0
    non_ground_point = []

    # 使用Ransac进行地面检测（迭代N次）
    for itr in range(max_iter):
        # Step1：随机选点（平面检测：3个样本点）
        #seed = np.random.sample(data, sample)
        seeds = random.sample(data.tolist(), sample) # list
        seeds_arr = np.asarray(seeds)
        # Step2：建立模型
        coef = estimate_normal(seeds_arr, normalize = True).reshape(-1,1)
        
        # Step3：计算error function for each point
        dis_points = np.abs(np.dot(data, coef[:3,:])+coef[3,:]) # 所有点到平面的距离矩阵
        # Step4：计算平面的点数（阈值：dis_t）
        plane_points_idx = [ir for ir in range(len(dis_points)) if dis_points[ir] < dis_t] # 得到地面点索引
        T = len(plane_points_idx)
        # Step5：选择最多内点的模型
        if T > max_point:
            max_point = T
            final_coef = coef # 得到最终模型参数（法向量）
            ground_point_idx = plane_points_idx
            non_ground_point_idx = [ir for ir in range(len(dis_points)) if dis_points[ir] >= dis_t]
            
        # 提前终止：
        if (max_point / N) > (1- e):
            break

    print('origin data points num:', N)
    print('segmented data points num:', len(non_ground_point_idx))
    return non_ground_point_idx, ground_point_idx

def getCore(data, r, min_samples, method = 'kdtree'):
    """Get core points from point clouds

    功能：用于寻找满足条件的核心点(在r范围内，数量大于或等于min_samples)
    可以使用欧式空间进行计算，也可以利用kdtree进行加速，快速搜索

    Args:
        data: input data
        r: 搜索范围
        min_samples: 最少点数量
        method: 使用的搜索策略

    Returns:
        core_idx(set): 核心点索引值集合
        CoreSet(dict): 核心点邻近点集合(包含自身)
    """
    def getNeibor(data, dataSet, r):
        '''在数据集dataSet中，以数据点data为中心，以r半径，获得满足条件的数据点，返回索引值
        
        Args:
            data: 1*3
            dataSet: N*3
            r: float 
        
        Returns:
            neibor: list, 邻域点的索引值
        '''
        neibor = []
        dis_matrix  = np.sum((data - dataSet)**2, axis = 1) # 中心点到数据集所有点的距离的平方
        for idx in range(dis_matrix.shape[0]):
            if dis_matrix[idx] < r*r:
                neibor.append(idx)
        return neibor

    core_idx = set() # set:核心点索引
    CoreSet = {} # dict:核心点邻近点集合
    if method == 'kdtree':
        tree = KDTree(data, leaf_size= 1) # leaf_size: 可以调整
        ind = tree.query_radius(data, r)
        for i in range(ind.shape[0]):
            if len(ind[i]) >= min_samples:
                core_idx.add(i)
                CoreSet[i] = ind[i]
    if method == 'euclidean':
        for i in range(data.shpae[0]):
            neibor = getNeibor(data[i], data, r) # 得到该数据点的邻域（索引）
            if  len(neibor) >= min_samples:
                core_idx.add(i) # 存储核心点索引
                CoreSet[i] = neibor
    return core_idx, CoreSet

def clustering(data, r = 0.4, min_samples = 5):
    '''从点云中提取聚类

    输入滤去地面后的点云，利用DBSCAN来进行聚类

    Args:
        data: 点云 n*3 （滤除地面后的点云）
        r: 搜索邻域的范围
        min_samples: 判断为核心点的邻域最小样本数
    
    Returns:
        clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号
    '''

    N,_ = data.shape
    # 初始化
    core_idx = set()
    CoreSet = {} # 核心点集合(字典)
    C = set() # 聚类集合
    clusters_index = np.zeros(N, dtype = int) 
    noise = [] 
    P_not_visit = set(range(N)) # 未访问集合(索引)
    k = 0 # 第k类

    # 找出所有核心点
    # 利用kdtree或者欧式距离，首先对聚类点集进行排序，然后query for neighbors within a given radius
    core_idx, CoreSet = getCore(data, r, min_samples, method = 'kdtree')

    # 遍历所有核心点
    while len(core_idx):
        P_old = P_not_visit
        core_rdn = list(core_idx)[np.random.randint(0,len(core_idx))] # 随机从核心点集合中选出一个核心点（索引）
        P_not_visit = P_not_visit - set([core_rdn])     # 将访问过的核心点从未访问集合中删去
        queue = []
        queue.append(core_rdn)
        # 找到不同的cluster
        while len(queue):
            q = queue[0]
            if q in core_idx:
                Connect = set(CoreSet[q]) # 找到该核心所有邻域点（索引）
                S = Connect & P_not_visit # 不包含核心点的所有邻域点
                queue += (list(S))
                #print(queue)
                P_not_visit = P_not_visit - S # 所有点减去连通域
            queue.remove(q)
        C = P_old - P_not_visit # 得到连通域相同的聚类集合
        core_idx = core_idx - C 
        k+=1
        clusters_index[list(C)] = k
        
    return clusters_index

def plot_clusters_o3d(data, cluster_index):
    '''使用o3d对聚类结果进行显示（待完成）
    '''
    def map(color_index, num_clusters):
        '''
        '''
        color = [0]*3
        color = [color_index/num_clusters] *3
        return color

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.5,0.5,0.5])  #给全部点云上灰色
    num_clusters = cluster_index.max() + 1
    pcd.colors = o3d.utility.Vector3dVector([
        map(label, num_clusters) for label in cluster_index
    ])
    # visualize
    o3d.visualization.draw_geometries([pcd])

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def plot_pt(data, index = None):
    '''显示点云数据
    输入点云坐标以及它的索引

    Args:
        data(np.array): 点云数据，N*3
        index(list): 需要显示的点云索引
    '''
    pcd = o3d.geometry.PointCloud()
    if index != None:
        pcd.points = o3d.utility.Vector3dVector(data[index])
    else:
        pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])

def main():
    root_dir = '/home/magictz/Projects/shenlan/3D_point_cloud/ch4_modelfitting/data' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        print('-------------显示原始点云----------------')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(origin_points)
        o3d.visualization.draw_geometries([pcd]) # 原始点云
        print('----------------去除地面-------------------')
        # 方法：Ransac
        segmented_points_idx, ground_point_idx = ground_segmentation(data=origin_points, dis_t = 0.25, e = 0.4)
        # 可视化地面点和非地面点
        plot_pt(origin_points,ground_point_idx) # 地面点
        plot_pt(origin_points,segmented_points_idx) # 非地面点
        
        print('-------------优化：采样------------')
        filter_points = filter.voxel_filter(origin_points[segmented_points_idx],leaf_size = 0.1) # 对原始聚类点云采样，降低数据的规模
        print('sampling data points num: ', filter_points.shape[0])
        plot_pt(filter_points) # 采样后的非地面点
        
        print('------------对非地面点云进行聚类-----------')
        # 原始数据直接处理
        segmented_points = origin_points[segmented_points_idx]
        start_t = time.time()
        cluster_index = clustering(segmented_points) # 原始数据进行聚类
        origin_t = time.time() - start_t
        # 对采样数据进行处理
        start_t = time.time()
        filter_cluster_index = clustering(filter_points) # 采样数据进行聚类
        filter_t = time.time() - start_t
        print('原始数据聚类时间：', origin_t)
        print('采样数据聚类时间：', filter_t)
        #plot_clusters_o3d(segmented_points, cluster_index)
        plot_clusters(segmented_points, cluster_index)
        #plot_clusters_o3d(filter_points, filter_cluster_index)
        plot_clusters(filter_points, filter_cluster_index)
        

if __name__ == '__main__':
    main()
