# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as  plt
from mpl_toolkits.mplot3d import Axes3D

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始

    # Normalized by the center
    N = np.size(data, 0) # the size of the matrix (row size)
    data_mean = np.mean(data, axis = 0) # [1*3]
    data_norm = data - data_mean # [N*3]

    # Compute Covariance matrix or Correlation coefficient matrix
    if correlation == False:
        H = np.corrcoef(data_norm, rowvar = False, bias = False)
    else:
        H = np.cov(data_norm, rowvar = False, bias = False) # H[3*3],以列为变量,标准化除以n-1

    # Compute SVD
    U, sigma, VT = np.linalg.svd(H) # U: eigenvector matrix; sigma: eigenvalue matrix
    eigenvectors = U            # [3*3] 
    eigenvalues = sigma     #[1*3]

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


# 功能:读入点云文件
# 输入:
#       filename: 文件名
#       Separator: 分隔符号
# 输出:
#       point[x,y,z]: 点云坐标数组
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
    print("读入点的个数为:{}个".format(num))
    point = [x, y, z]
    return point

# 功能: 显示三维点云
def displayCloud(cloud):
    #开始绘图
    fig=plt.figure(dpi=120)
    ax = Axes3D(fig)
    #标题
    ax.set_title('point cloud')
    #利用xyz的值，生成每个点的相应坐标（x,y,z）
    ax.scatter(cloud[0],cloud[1],cloud[2],c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #显示
    plt.show()


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    abs_path = os.path.abspath(os.path.dirname(__file__)) # 获取当前文件绝对路径
    filename = os.path.join(abs_path, 'car_0001.txt') # 数据集文件路径
    Separator = ',' # 分隔符
    point_cloud = readXYZfile(filename, Separator)
    displayCloud(point_cloud)
    array = np.array(point_cloud)


    # 加载原始点云
    cloud = PyntCloud(array)
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    # w: eigenvalues
    # v: eigenvector
    w, v = PCA(points)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
