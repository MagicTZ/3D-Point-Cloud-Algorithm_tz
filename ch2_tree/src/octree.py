# octree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet

# 节点，构成OCtree的基本元素
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output

# 功能：翻转octree
# 输入：
#     root: 构建好的octree
#     depth: 当前深度
#     max_depth：最大深度
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1

# 功能：通过递归的方式构建octree
# 输入：
#     root：根节点
#     db：原始数据
#     center: 中心
#     extent: 当前分割区间
#     point_indices: 点的key
#     leaf_size: scale
#     min_extent: 最小分割区间
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        # 作业4
        # 屏蔽开始
        root.is_leaf = False
        child_point_idxs = [[] for i in range(8)] # 定义区域并且设置大小，否则后面for循环无法循环添加元素
        for point_idx in point_indices:
            point_db = db[point_idx]
            child_idx = 0
            # 判断数据点的位置（与中心点比较）
            if point_db[0] > center[0]:
                child_idx = child_idx | 1   # 分辨x
            if point_db[1] > center[1]:
                child_idx = child_idx | 2   # 分辨y
            if point_db[2] > center[2]:
                child_idx = child_idx | 4   # 分辨z

            child_point_idxs[child_idx].append(point_idx) # 将点的放到8个区域中对应的位置

        factor = [-0.5, 0.5]
        for i in range(8):
            # 重新计算得到各个区域的中心
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = extent * 0.5 # 重新计算1/2边长
            child_center  = np.array([child_center_x, child_center_y, child_center_z])
            root.children[i] = octree_recursive_build(root.children[i],
                                                                                                db,
                                                                                                child_center,
                                                                                                child_extent,
                                                                                                child_point_idxs[i],
                                                                                                leaf_size,
                                                                                                min_extent)

        return root        
        # 屏蔽结束
    return root

# 功能：判断当前query区间是否在octant内
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)

# 功能：判断当前query区间是否和octant有重叠部分
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_center_dis = query - octant.center # 点到区域中心的距离
    query_center_dis_abs = np.fabs(query_center_dis)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_center_dis_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_center_dis_abs < octant.extent).astype(np.int)) >= 2: # 至少有两个维度在区域范围
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_center_dis_abs[0] - octant.extent, 0)
    y_diff = max(query_center_dis_abs[1] - octant.extent, 0)
    z_diff = max(query_center_dis_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


# 功能：判断当前query是否包含octant
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def contains(query: np.ndarray, radius: float, octant:Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius

# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 作业5
    # 提示：尽量利用上面的inside、overlaps、contains等函数
    # 屏蔽开始

    # 直接在octree_radius_search()里修改了
    
    # 屏蔽结束

    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找radius范围内的近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray, search = ''):
    if root is None:
        return False

    # 另外一种极端情况：查询半径已经包括了整个区域
    if search == 'fast':
        if contains(query, result_set.worstDist(), root):
        
            leaf_points = db[root.point_indices, :]
            diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
            for i in range(diff.shape[0]):
                result_set.add_point(diff[i], root.point_indices[i])
            return False

    # 如果是叶子，则将该区域内返回到result_set
    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业6
    # 屏蔽开始
    child_idx = 0
    # 判断所查询点在八叉树的位置
    if query[0] > root.center[0]:
        child_idx = child_idx | 1
    if query[1] > root.center[1]:
        child_idx = child_idx | 2
    if query[2] > root.center[2]:
        child_idx = child_idx | 4
    
    # 递归判断是否在该区域就可以找到足够多的点
    if octree_radius_search(root.children[child_idx], db, result_set, query):
        return True

    # 没有在查询点区域找到，对其他区域进行搜索
    for c, child in enumerate(root.children):
        if c == child_idx or child is None: # 搜索区域没有点或所搜索到查询点区域，skip
            continue
        if False == overlaps(query, result_set.worstDist(), child): # 搜索区域与查询点和最坏距离构成的球面没有交点，skip
            continue
        if octree_radius_search(child, db, result_set, query): # 其他情况可以进入搜索区域搜索（递归）
            return True
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)

# 功能：在octree中查找最近的k个近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业7
    # 屏蔽开始
    child_idx = 0
    # 判断所查询点在八叉树的位置
    if query[0] > root.center[0]:
        child_idx = child_idx | 1
    if query[1] > root.center[1]:
        child_idx = child_idx | 2
    if query[2] > root.center[2]:
        child_idx = child_idx | 4
    
    # 递归判断是否在该区域就可以找到足够多的点
    if octree_knn_search(root.children[child_idx], db, result_set, query):
        return True

    # 没有在查询点区域找到，对其他区域进行搜索
    for c, child in enumerate(root.children):
        if c == child_idx or child is None: # 搜索区域没有点或所搜索到查询点区域，skip
            continue
        if False == overlaps(query, result_set.worstDist(), child): # 搜索区域与查询点和最坏距离构成的球面没有交点，skip
            continue
        if octree_knn_search(child, db, result_set, query): # 其他情况可以进入搜索区域搜索（递归）
            return True
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)

# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
# 输入：
#    dp_np: 原始数据
#    leaf_size：scale
#    min_extent：最小划分区间
def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1] # 点数，维度
    db_np_min = np.amin(db_np, axis=0) # 1*3 找到最小(x,y,z)
    db_np_max = np.amax(db_np, axis=0) # 1*3 找到最大(x,y,z)
    db_extent = np.max(db_np_max - db_np_min) * 0.5 # 计算1/2边长
    db_center = np.mean(db_np, axis=0) # 中心点

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent) # 循环建树

    return root

def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    # depth = [0]
    # max_depth = [0]
    # traverse_octree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # octree_knn_search(root, db_np, result_set, query)
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])

    # 测试octree_radius_search（Normal）
    begin_t = time.time()
    print("Radius search normal:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.5)
        octree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    # 测试radius_search (fast)
    begin_t = time.time()
    print("Radius search fast:")
    for i in range(100):
        query = np.random.rand(3) # 这里有点小问题：查找点应该保持相同(deep.copy)
        result_set = RadiusNNResultSet(radius = 0.5)
        #octree_radius_search_fast(root, db_np, result_set, query)
        octree_radius_search(root, db_np, result_set, query, search = 'fast')
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t)*1000))

    # 测试knn_search
    begin_t = time.time()
    print("KNN search :")
    for i in range(100):
        query = np.random.rand(3) # 这里有点小问题：查找点应该保持相同(deep.copy)
        result_set = KNNResultSet(capacity = k)
        octree_knn_search(root, db_np, result_set, query)
    print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t)*1000))



if __name__ == '__main__':
    main()