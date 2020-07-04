# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能:读入点云文件
# 输入:
#       filename: 文件名
#       Separator: 分隔符号, default = " "
# 输出:
#       point(np.array): 点云数据, N*6
def readXYZfile(filename, Separator = " "):
    data = [[],[],[],[],[],[]]
    
    num = 0
    for line in open(filename, 'r'): #按行读入点云
        line = line.strip('\n') # 去掉换行符
        a,b,c,d,e,f = line.split(Separator)
        data[0].append(a) #X坐标
        data[1].append(b) #Y坐标
        data[2].append(c) #Z坐标
        data[3].append(d)
        data[4].append(e)
        data[5].append(f)
        num = num + 1

    #string to float
    x = [float(data[0]) for data[0] in data[0]]
    y = [float(data[1]) for data[1] in data[1]]
    z = [float(data[2]) for data[2] in data[2]]
    nx = [float(data[3]) for data[3] in data[3]]
    ny = [float(data[4]) for data[4] in data[4]]
    nz = [float(data[5]) for data[5] in data[5]]
    print("读入点的个数为:{}个".format(num))
    point = [x, y, z, nx, ny, nz]
    point = np.array(point) # list to np.array 

    point = point.transpose() # 6*N to N*6
    return point


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
#     method: downsample method, centroid or random, default: centroid
def voxel_filter(point_cloud, leaf_size, method = 'centroid'):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # 获取点云的bbox的范围
    x_max = np.max(point_cloud[:,0], axis = 0)
    x_min = np.min(point_cloud[:,0], axis =0)
    y_max = np.max(point_cloud[:,1], axis =0)
    y_min = np.min(point_cloud[:,1], axis = 0)
    z_max = np.max(point_cloud[:,2], axis =0)
    z_min = np.min(point_cloud[:,2], axis =0)

    # Compute the dimension of the voxel grid
    Dx = ((x_max - x_min)/leaf_size).astype(np.int)
    Dy = ((y_max - y_min)/leaf_size).astype(np.int)
    Dz = ((z_max - z_min)/leaf_size).astype(np.int)

    # Compute voxel index for each point
    hx = ((point_cloud[:,0]- x_min)/leaf_size).astype(np.int)
    hy = ((point_cloud[:,1]- y_min)/leaf_size).astype(np.int)
    hz = ((point_cloud[:,2]- z_min)/leaf_size).astype(np.int)
    idx = np.dtype(np.int64)
    idx= hx + hy*Dx + hz*Dx*Dy # 得到每一个点的索引值

    point_cloud_idx = np.insert(point_cloud, 0, values = idx, axis = 1) # 将索引值与点云数据合并
    #point_cloud_idx = np.c_[idx, point_cloud] # 合并方法二

    # Sort by the index
    point_cloud_idx = point_cloud_idx[np.lexsort(point_cloud_idx[:,::-1].T)]
    #print(point_cloud_idx[0:15,:])

    # Select points according to centroid/random method
    point_cloud_idx[:,0].astype(np.int)
    n = 0
    k = point_cloud_idx[0,0]
    if method == 'centroid':
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                # print(np.mean(point_cloud_idx[n:i, :], axis = 0))
                filtered_points.append(np.mean(point_cloud_idx[n:i,1:4], axis =0))
                k = point_cloud_idx[i,0]
                n = i
    elif method == 'random':
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                # print(np.mean(point_cloud_idx[n:i, :], axis = 0))
                point_rand = np.random.randint(n,i) # 在[n, i) 范围内随机选出一个点作为采样点
                filtered_points.append(point_cloud_idx[point_rand,1:4])
                k = point_cloud_idx[i,0]
                n = i

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 指定点云路径
    #abs_path = os.path.abspath(os.path.dirname(__file__)) # 获取当前文件绝对路径
    #filename = os.path.join(abs_path, 'car_0001.txt') # 数据集文件路径

    root_dir = '/home/magictz/Projects/shenlan/dataset/modelnet40_normal_resampled'
    filenames = os.path.join(root_dir, 'modelnet40_shape_names.txt')
    filename = []
    # 根据name文件,逐行读取40个model并显示
    for line in open(filenames, 'r'):
        line = line.strip('\n')
        filename =  os.path.join(root_dir, line, line+'_0001.txt')
        # 从TXT文件中获取点云信息
        pointcloud = readXYZfile(filename, Separator= ',') 

        # 加载原始点云(利用open3d)
        # 方法一: 利用pyntcloud直接读取.txt并创建点云对象
        # pointcloud_pynt = PyntCloud.from_file(filename, 
        #                                                         sep=",", 
        #                                                         header =-1, 
        #                                                         names = ["x", "y", "z"])
        # pcd = pointcloud_pynt.to_instance("open3d", mesh = False)
        # 方法二: 利用自定义readXYZfile()
        pcd = o3d.geometry.PointCloud() # 创建一个pointcloud对象pcd
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:,0:3]) # 读入点云数据的x,y,z
        pcd2 = o3d.geometry.PointCloud()
        o3d.visualization.draw_geometries([pcd])
        
        # 调用voxel滤波函数，实现滤波
        filtered_cloud = voxel_filter(pointcloud[:,0:3], 0.07, method= 'random') # voxel grid resolution: 0.1m
        filtered_cloud_c = voxel_filter(pointcloud[:,0:3], 0.07, method= 'centroid') # voxel grid resolution: 0.1m
        pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
        pcd2.points = o3d.utility.Vector3dVector(filtered_cloud_c)


        # 显示滤波后的点云
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([pcd2])



if __name__ == '__main__':
    main()
