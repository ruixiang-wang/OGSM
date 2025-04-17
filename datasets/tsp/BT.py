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


# 回溯算法求解TSP问题
def backtracking(nodes):
    n = len(nodes)
    best_path = None
    best_distance = float('inf')

    # 用于回溯的辅助函数
    def dfs(path, visited, current_distance):
        nonlocal best_path, best_distance
        if len(path) == n:
            # 如果路径包含所有节点，计算回到起点的距离并更新最短路径
            total_distance = current_distance + euclidean_distance(nodes[path[-1]], nodes[path[0]])
            if total_distance < best_distance:
                best_distance = total_distance
                best_path = path.copy()
            return

        # 递归尝试访问下一个节点
        for next_node in range(n):
            if not visited[next_node]:
                # 选择当前节点，加入路径
                visited[next_node] = True
                path.append(next_node)

                # 递归进行下一步搜索
                dfs(path, visited, current_distance + (euclidean_distance(nodes[path[-2]], nodes[next_node]) if len(path) > 1 else 0))

                # 回溯，撤销当前选择
                visited[next_node] = False
                path.pop()

    # 从起点开始DFS搜索
    for start_node in range(n):
        visited = [False] * n
        visited[start_node] = True
        dfs([start_node], visited, 0)

    return best_path, best_distance


# 计算路径的总距离
def calculate_total_distance(nodes, path):
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += euclidean_distance(nodes[path[i]], nodes[path[i + 1]])
    total_distance += euclidean_distance(nodes[path[-1]], nodes[path[0]])  # 回到起点
    return total_distance


# 输出结果
def output_result(filename, path, total_distance, optimal_distance):
    print(f"File: {filename} - Shortest Path: {path} - Total Distance: {total_distance:.6f} - Difference: {abs(total_distance - optimal_distance) / optimal_distance * 100:.2f}%")


# 遍历指定目录下的所有TSP文件并求解
def solve_all_tsp_in_directory(directory, optimal_solutions):
    for filename in os.listdir(directory):
        if filename.endswith('.tsp'):
            filepath = os.path.join(directory, filename)
            nodes = read_tsp_file(filepath)
            path, total_distance = backtracking(nodes)
            if filename in optimal_solutions:
                optimal_distance = optimal_solutions[filename]
                output_result(filename, path, total_distance, optimal_distance)
            else:
                print(f"Optimal solution for {filename} not found.")


def read_optimal_solutions(filename):
    optimal_solutions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_name = line.split(":")[0].strip()  # 去掉 .tsp 后缀
            total_distance = float(line.split("Total Distance: ")[1].strip())
            optimal_solutions[file_name] = total_distance
    return optimal_solutions


# 主函数
if __name__ == "__main__":
    directory = './create_problem'  # 创建问题的文件夹路径
    optimal_solutions = read_optimal_solutions('optimal_solutions.txt')
    solve_all_tsp_in_directory(directory, optimal_solutions)
