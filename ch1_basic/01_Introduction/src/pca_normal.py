# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as  plt
from mpl_toolkits.mplot3d import Axes3D


def PCA(data, correlation=False, sort=True):
    ''' PCA implementation function

    Args:
        data: point cloud (N*3 matrix).
        correlation: A boolean indicating if we use covariance matrix or
            correlation coefficient matrix (default: correlation coefficient 
            matrix).
        sort: A boolean indicating if we sort by eigenvalue.

    Returns:
        eigenvalues: The eigenvalues computed by SVD
        eigenvectors: The eigenvectors computed by SVD 
    '''
    # Normalized by the center
    N = np.size(data, 0) # the size of the matrix (row size)
    data_mean = np.mean(data, axis = 0) # [1*3]
    data_norm = data - data_mean.T # [N*3]

    # Compute Covariance matrix or Correlation coefficient matrix
    if correlation == True:
        H = np.corrcoef(data_norm, rowvar = False, bias = False)
    else:
        H = np.cov(data_norm, rowvar = False, bias = True) # H[3*3],以列为变量,标准化除以n-1

    # Compute SVD
    U, sigma, VT = np.linalg.svd(H) # U: eigenvector matrix; sigma: eigenvalue matrix
    eigenvectors = U            # [3*3] 
    eigenvalues = sigma     #[1*3]

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def readXYZfile(filename, Separator = " "):
    ''' Read the point cloud file

    Args:
        filename: A string representing point cloud file path.
        Separator: A charactor used to separate words in every line.
    
    Returns:
        point: An array representing point cloud data (N*6 matrix).
    '''
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

def displayCloud(cloud):
    ''' Display point cloud

    Args:
        cloud: An 3*N array indicating point cloud data
    '''

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

# 功能: 将TXT格式的点云文件转成ply格式
# 输入:
#       filename: 文件名
#       path: 绝对路径
#       pointcloud: 点云数组, N*3
def create_ply(pointcloud, filename, path):
    savefilepath = os.path.join(path, filename)
    pointcloud = pointcloud.reshape(-1,6)
    np.savetxt(savefilepath, pointcloud, fmt = '%f %f %f %f %f %f')
    # 利用write() 在头部插入ply_header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''

    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header%dict(vert_num = len(pointcloud)))
        f.write(old)

# 功能:根据PCA得到的特征向量变换坐标基并显示
# 输入:
#       pointcloud: 原始点云数组, N*3
#       vector: 特征向量
# 输出:
#       变换后的点云坐标
def pcl_decoder(pointcloud, vector):
    '''
    mean = np.mean(pointcloud, axis = 0)
    points = [
            [vector[:,0]],
            [vector[:,1]],
            [vector[:,2]],
            [0, 0, 0],
    ]
    lines = [
            [0,3],
            [1,3],
            [2,3],
    ]
    colors = [
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ]
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(vector),
        lines = o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    '''
    pointcloud_new  = np.dot(pointcloud, vector)

    pcd = o3d.geometry.PointCloud() # 创建一个pointcloud对象(经过PCL转换后的数据)
    pcd.points = o3d.utility.Vector3dVector(pointcloud_new) # 读入点云数据的x,y,z
    o3d.visualization.draw_geometries([pcd]) # 可视化
    return pointcloud_new

# 功能: 利用PCA方法和knn循环计算点云中每个点的法向量
# 输入: 
#       pcd: PointCloud对象
#       iter_n: 迭代次数(点云中点的个数)
#       knn: 邻近点个数
# 输出:
#       normals: 法向量
def get_normal(pcd, iter_n, knn):
    # Build KDTree from the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals =[]
    for i in range(iter_n):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)  
        nearest_points = np.asarray(pcd.points)[idx,:]
        w, v = PCA(nearest_points)
        normals.append(v[:,2])
    
    return normals


def main():
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
    

        # Create point cloud file ".ply" (如果需要.ply文件可以使用下面的代码)
        # plyfile = 'car_0001.ply'
        # create_ply(array.transpose(), plyfile, abs_path)
        # filename = os.path.join(abs_path, plyfile)

        # 加载原始点云(利用open3d)
        # 方法一: 利用pyntcloud直接读取.txt并创建点云对象
        # pointcloud_pynt = PyntCloud.from_file(filename, 
        #                                                         sep=",", 
        #                                                         header =-1, 
        #                                                         names = ["x", "y", "z"])
        # pcd = pointcloud_pynt.to_instance("open3d", mesh = False)
        # 方法二: 利用自定义readXYZfile()来读取文件
        pointcloud = readXYZfile(filename, Separator = ",")
        pcd = o3d.geometry.PointCloud() # 创建一个pointcloud对象pcd
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:,0:3]) # 读入点云数据的x,y,z
        o3d.visualization.draw_geometries([pcd]) # 显示原始点云

        # 从点云中获取点，只对点进行处理
        points = pcd.points 
        print('total points number is:', pointcloud.shape[0])
    

        # 用PCA分析点云主方向
        # w: eigenvalues
        # v: eigenvector
        w, v = PCA(points)
        point_cloud_vector = v[:, 2] #点云主方向对应的向量
        print('the main orientation of this pointcloud is: ', point_cloud_vector)
        # Decoder: 根据不同成分来还原point cloud
        new_vector = np.zeros((3,3))
        new_vector2 = np.zeros((3,3))
        new_vector3 = np.zeros((3,3))
        new_vector[:, 0] = v[:,0] # 仅保留主向量
        new_vector2[:, 0:2] = v[:,0:2] # 保留第一和第二分量
        new_vector3[:,:] = v[:,:] #保留所有分量
        point_cloud_new = pcl_decoder(points, new_vector)
        point_cloud_new2 = pcl_decoder(points, new_vector2)
        point_cloud_new3 = pcl_decoder(points, new_vector3)
    
        # 在原始点云中画出不同分量的方向
        # 待实现....
    
        # 循环计算每个点的法向量
        # 作业2
        # 屏蔽开始
        iter_n = pointcloud.shape[0] # 原始点云数(即循环次数)
        knn_n = 5 # 邻近点个数
        normals = get_normal(pcd, iter_n, knn_n)

        # 屏蔽结束
        normals = np.array(normals, dtype=np.float64)
        # TODO: 此处把法向量存放在了normals中
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd]) # 按住'n'查看法向量
    
        # 显示法向量
        # 待实现....
    


if __name__ == '__main__':
    main()
