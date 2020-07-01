# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能:读入点云文件
# 输入:
#       filename: 文件名
#       Separator: 分隔符号
# 输出:
#       point[x,y,z]: 点云坐标list
def readXYZfile(filename, Separator):
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
    r = [float(data[3]) for data[3] in data[3]]
    g = [float(data[4]) for data[4] in data[4]]
    b = [float(data[5]) for data[5] in data[5]]
    print("读入点的个数为:{}个".format(num))
    point = [x, y, z, r, g, b]
    return point


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
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
    Dx = np.int64((x_max - x_min)/leaf_size)
    Dy = np.int64((y_max - y_min)/leaf_size)
    Dz = np.int64((z_max - z_min)/leaf_size)

    # Compute voxel index for each point
    hx = np.int64(point_cloud[:,0]- x_min)/leaf_size
    hy = np.int64(point_cloud[:,1]- y_min)/leaf_size
    hz = np.int64(point_cloud[:,2]- z_min)/leaf_size
    idx = np.dtype(np.int64)
    idx= hx + hy*Dx + hz*Dx*Dy

    # Sort by the index
    

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
    abs_path = os.path.abspath(os.path.dirname(__file__)) # 获取当前文件绝对路径
    filename = os.path.join(abs_path, 'car_0001.txt') # 数据集文件路径
    
    # 从TXT文件中获取点云信息
    Separator = ',' # 分隔符
    pointcloud = readXYZfile(filename, Separator)
    # displayCloud(point_cloud) # 利用matplotlib画出点云
    pointcloud = np.array(pointcloud) # List to np.array
    pointcloud = pointcloud.transpose()
    print(pointcloud.shape)

    # 加载原始点云(利用open3d)
    pcd = o3d.geometry.PointCloud() # 创建一个pointcloud对象pcd
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:,0:3]) # 读入点云数据的x,y,z

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(pcd.points, 100.0)
    pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
