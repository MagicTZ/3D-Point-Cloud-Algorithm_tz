# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
import random

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

from scipy import spatial # 内含kdtree

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content) # 对二进制文件循环解包并返回索引
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.005
    k = 8
    radius = 1
    nums = 1# 点的数量（搜索）

    abs_path = os.path.abspath(os.path.dirname(__file__)) # 数据集路径 (数据集直接放在当前路径下)
    filename = os.path.join(abs_path, '000000.bin') # 只读取一个文件,如果要读取所有文件,需要循环读入
    iteration_num = 1

    # 读取数据并进行采样
    db_np_origin = read_velodyne_bin(filename) # N*3
    db_np_idx = np.random.choice(db_np_origin.shape[0],size =(30000,)) # 随机采样30000个点
    db_np = db_np_origin[db_np_idx]

    #root_dir = '/Users/renqian/cloud_lesson/kitti' # 数据集路径
    #cat = os.listdir(root_dir)
    #iteration_num = len(cat)

    print("搜索数据集中的 %d 个点: " % nums)
    print("---------------------------------Octree ----------------------------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num): 
    #filename = os.path.join(root_dir, cat[i])
        '''
        db_np_origin = read_velodyne_bin(filename) # N*3
        db_np_idx = np.random.choice(db_np_origin.shape[0],size =(30000,)) # 随机采样30000个点
        db_np = db_np_origin[db_np_idx]
        '''
        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)     # Octree Construction
        construction_time_sum += time.time() - begin_t

        print("The sum of the points: ", db_np.shape[0])
        nums = db_np.shape[0]
        # 查找nums个点
        for num in range(nums):
         
            query = db_np[num,:] # 搜索点数组（假设第一个点为搜索点）

            # knn search
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            octree.octree_knn_search(root, db_np, result_set, query) 
            knn_time_sum += time.time() - begin_t

            # radius search
            begin_t = time.time()
            result_set = RadiusNNResultSet(radius=radius)
            #octree.octree_radius_search_fast(root, db_np, result_set, query)
            octree.octree_radius_search(root, db_np, result_set, query, search='fast')
            radius_time_sum += time.time() - begin_t

            # brute search
            begin_t = time.time()
            diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
            nn_idx = np.argsort(diff)
            nn_dist = diff[nn_idx]
            brute_time_sum += time.time() - begin_t

        print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                        knn_time_sum*1000/iteration_num,
                                                                        radius_time_sum*1000/iteration_num,
                                                                        brute_time_sum*1000/iteration_num))

    print("---------------------------------Kdtree ----------------------------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    knn_scipy_time_sum = 0 
    radius_scipy_time_sum = 0
    for i in range(iteration_num):
        #filename = os.path.join(root_dir, cat[i])
        #db_np = read_velodyne_bin(filename) # N*3

        print("The sum of the points: ", db_np.shape[0])
        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size) # kd tree construction
        construction_time_sum += time.time() - begin_t

        tree = spatial.KDTree(db_np) # 利用spatial库函数构造kdtree

        for num in range(nums):
            query = db_np[num,:]

            # scipy.spatial.KDtree
            begin_t = time.time()
            tree.query(query, k=k)
            knn_scipy_time_sum += time.time() - begin_t
            #print(tree.query(query, k=k))

            # knn search
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            kdtree.kdtree_knn_search(root, db_np, result_set, query)
            knn_time_sum += time.time() - begin_t
            #print(result_set)

            # scipy.spatial.KDtree (radius search)
            begin_t = time.time()
            tree.query_ball_point(query, radius)
            radius_scipy_time_sum += time.time() - begin_t

            # radius search
            begin_t = time.time()
            result_set = RadiusNNResultSet(radius=radius)
            kdtree.kdtree_radius_search(root, db_np, result_set, query)
            radius_time_sum += time.time() - begin_t
           # print(result_set)

            # brute search
            begin_t = time.time()
            diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
            nn_idx = np.argsort(diff)
            nn_dist = diff[nn_idx]
            brute_time_sum += time.time() - begin_t

        print("Kdtree: build %.3f, knn_scipy %.3f, knn %.3f, radius_scipy %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_scipy_time_sum*1000/iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_scipy_time_sum *1000 /iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))

    

if __name__ == '__main__':
    main()