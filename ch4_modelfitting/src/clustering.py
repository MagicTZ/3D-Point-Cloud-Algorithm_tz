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

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def estimate_normal(point, normalize = True):
    '''
    已知3个平面点point, 求平面方程，返回系数矩阵
    :param point: 3*3 array
    :param normalize: bool
    :return coef:  1*4 array
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
    

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云, N*3
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, sample = 3, max_iter = 100, dis_t = 0.5):
    # 作业1
    # 屏蔽开始
    N, D = data.shape # N：总点数， D：维度
    T = 0 # 地面点数量
    max_point = 0
    e = 0.3 # 非地面点占所有点的概率
    non_ground_point = []
    dis_t = 0.3 # 判断地面点阈值

    max_iter = 30   # 最大迭代次数
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
            final_coef = coef
            ground_point_idx = plane_points_idx
            non_ground_point_idx = [ir for ir in range(len(dis_points)) if dis_points[ir] >= dis_t]
            

        # 提前终止：
        if (max_point / N) > (1- e):
            break
        
    return non_ground_point_idx, ground_point_idx
    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始


    # 屏蔽结束

    return clusters_index

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

def main():
    root_dir = '/home/magictz/Projects/shenlan/3D_point_cloud/ch4_modelfitting/src' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[:]
    #iteration_num = len(cat)
    iteration_num = 1

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        print('-------------显示原始点云----------------')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(origin_points)
        #o3d.visualization.draw_geometries([pcd])
        print('----------------去除地面-------------------')
        # 方法：Ransac
        segmented_points_idx, ground_point_idx = ground_segmentation(data=origin_points)
        # 可视化地面点和非地面点
        pcd.points = o3d.utility.Vector3dVector(origin_points[ground_point_idx])
        o3d.visualization.draw_geometries([pcd])
        pcd.points = o3d.utility.Vector3dVector(origin_points[segmented_points_idx])
        o3d.visualization.draw_geometries([pcd])
        print('------------对其他点云进行聚类-----------')
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
