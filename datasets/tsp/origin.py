import os
import math
import random

# 计算两点之间的欧几里得距离
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 读取TSP文件，解析出节点的坐标
def read_tsp_file(filename):
    nodes = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == "NODE_COORD_SECTION":
                break
        # 读取坐标数据
        for line in lines[lines.index("NODE_COORD_SECTION\n") + 1:]:
            parts = line.split()
            if len(parts) == 3:
                nodes.append((float(parts[1]), float(parts[2])))
    return nodes


# 使用最邻近算法解决TSP问题
def nearest_neighbor(nodes):
    n = len(nodes)
    visited = [False] * n
    path = []
    total_distance = 0.0

    # 从第一个节点开始
    current_node = 0
    visited[current_node] = True
    path.append(current_node)

    while len(path) < n:
        nearest_node = None
        nearest_dist = float('inf')

        # 查找最近的未访问节点
        for i in range(n):
            if not visited[i]:
                dist = euclidean_distance(nodes[current_node], nodes[i])
                if dist < nearest_dist:
                    nearest_node = i
                    nearest_dist = dist

        # 更新当前节点、路径和总距离
        visited[nearest_node] = True
        path.append(nearest_node)
        total_distance += nearest_dist
        current_node = nearest_node

    # 完成一圈后返回起点
    total_distance += euclidean_distance(nodes[current_node], nodes[path[0]])

    return path, total_distance


# 使用最近插入法（NI）解决TSP问题
def nearest_insertion(nodes):
    n = len(nodes)
    visited = [False] * n
    path = []
    total_distance = 0.0

    # 从第一个节点开始（可以随机选择起始节点）
    start_node = 0
    path.append(start_node)
    visited[start_node] = True

    # 一开始，路径只有一个节点
    while len(path) < n:
        # 寻找下一个要插入的节点
        min_increase = float('inf')
        best_node = None
        best_position = None

        # 遍历当前路径的每对相邻节点
        for i in range(len(path)):
            for j in range(n):
                # 跳过已经访问过的节点
                if visited[j]:
                    continue

                # 计算插入节点的距离增加量
                node_i = path[i]
                node_j = path[(i + 1) % len(path)] if i + 1 < len(path) else path[0]
                dist_increase = (euclidean_distance(nodes[node_i], nodes[j]) +
                                 euclidean_distance(nodes[node_j], nodes[j]) -
                                 euclidean_distance(nodes[node_i], nodes[node_j]))

                # 选择增加距离最小的插入方式
                if dist_increase < min_increase:
                    min_increase = dist_increase
                    best_node = j
                    best_position = (i + 1) % len(path)

        # 插入最佳节点
        path.insert(best_position, best_node)
        visited[best_node] = True
        total_distance += min_increase

    # 计算路径的总距离
    total_distance = 0.0
    for i in range(len(path)):
        total_distance += euclidean_distance(nodes[path[i]], nodes[path[(i + 1) % len(path)]])

    return path, total_distance

# 使用随机插入法解决TSP问题
def random_insertion(nodes):
    n = len(nodes)
    visited = [False] * n
    path = []
    total_distance = 0.0

    # 随机选择一个起始节点
    current_node = random.randint(0, n - 1)
    visited[current_node] = True
    path.append(current_node)

    while len(path) < n:
        # 从未访问的节点中随机选择一个节点
        unvisited_nodes = [i for i in range(n) if not visited[i]]
        next_node = random.choice(unvisited_nodes)
        visited[next_node] = True

        # 将随机选择的节点插入路径
        # 在路径的任意位置插入新节点，这里我们插入到路径末尾
        path.append(next_node)

        # 更新路径总距离
        if len(path) > 1:
            total_distance += euclidean_distance(nodes[path[-2]], nodes[path[-1]])

    # 计算回到起始节点的距离
    total_distance += euclidean_distance(nodes[path[-1]], nodes[path[0]])

    return path, total_distance

# 使用最远插入法解决TSP问题
def farthest_insertion(nodes):
    n = len(nodes)
    visited = [False] * n
    path = []
    total_distance = 0.0

    # 随机选择起始节点
    current_node = random.randint(0, n-1)
    visited[current_node] = True
    path.append(current_node)

    # 选择第二个节点
    farthest_node = None
    max_dist = -1

    for i in range(n):
        if not visited[i]:
            dist = euclidean_distance(nodes[current_node], nodes[i])
            if dist > max_dist:
                max_dist = dist
                farthest_node = i

    visited[farthest_node] = True
    path.append(farthest_node)

    # 插入剩下的节点
    while len(path) < n:
        max_increase = -1
        node_to_insert = None
        insert_after = None

        # 遍历所有未访问的节点，找到最远插入的节点
        for i in range(n):
            if not visited[i]:
                # 计算将节点i插入路径中的每一个位置后的距离增量
                for j in range(len(path)):
                    if j == 0:
                        dist = euclidean_distance(nodes[i], nodes[path[j]]) + euclidean_distance(nodes[i], nodes[path[-1]]) - euclidean_distance(nodes[path[j]], nodes[path[-1]])
                    else:
                        dist = euclidean_distance(nodes[i], nodes[path[j]]) + euclidean_distance(nodes[i], nodes[path[j-1]]) - euclidean_distance(nodes[path[j]], nodes[path[j-1]])

                    if dist > max_increase:
                        max_increase = dist
                        node_to_insert = i
                        insert_after = j

        # 插入选择的节点
        path.insert(insert_after, node_to_insert)
        visited[node_to_insert] = True

    # 计算总距离
    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += euclidean_distance(nodes[path[i-1]], nodes[path[i]])

    total_distance += euclidean_distance(nodes[path[-1]], nodes[path[0]])  # 回到起点

    return path, total_distance

# 输出结果
def output_result(filename, path, total_distance, optimal_distance):
    print(f"File: {filename} - Shortest Path: {path} - Total Distance: {total_distance:.6f} - Difference: {abs(total_distance - optimal_distance) / optimal_distance * 100:.2f}%")

def read_optimal_solutions(filename):
    optimal_solutions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_name = line.split(":")[0].strip()  # 去掉 .tsp 后缀
            total_distance = float(line.split("Total Distance: ")[1].strip())
            optimal_solutions[file_name] = total_distance
    return optimal_solutions

# 遍历指定目录下的所有TSP文件并求解
def solve_all_tsp_in_directory(directory, optimal_solutions):
    for filename in os.listdir(directory):
        if filename.endswith('.tsp'):
            filepath = os.path.join(directory, filename)
            nodes = read_tsp_file(filepath)
            # path, total_distance = nearest_neighbor(nodes)
            # path, total_distance = nearest_insertion(nodes)
            # path, total_distance = random_insertion(nodes)
            path, total_distance = farthest_insertion(nodes)
            if filename in optimal_solutions:
                optimal_distance = optimal_solutions[filename]
                output_result(filename, path, total_distance, optimal_distance)
            else:
                print(f"Optimal solution for {filename} not found.")


# 主函数
if __name__ == "__main__":
    directory = './create_problem'  # 创建问题的文件夹路径
    optimal_solutions = read_optimal_solutions('optimal_solutions.txt')
    solve_all_tsp_in_directory(directory, optimal_solutions)
