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


# 蚁群算法求解TSP问题
# 蚁群算法求解TSP问题
def ant_colony_optimization(nodes, num_ants=10, num_iterations=100, alpha=1, beta=5, evaporation_rate=0.5,
                            pheromone_deposit=100):
    n = len(nodes)
    distances = [[euclidean_distance(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]

    # 初始化信息素矩阵
    pheromones = [[1.0 for _ in range(n)] for _ in range(n)]

    best_path = None
    best_distance = float('inf')

    for iteration in range(num_iterations):
        all_paths = []
        all_distances = []

        # 每只蚂蚁独立构建一个完整的路径
        for ant in range(num_ants):
            path = []
            visited = [False] * n
            current_node = random.randint(0, n - 1)
            path.append(current_node)
            visited[current_node] = True

            # 构建路径
            while len(path) < n:
                probabilities = []
                total_pheromone = 0.0
                for next_node in range(n):
                    if not visited[next_node]:
                        # 计算转移概率
                        pheromone = pheromones[current_node][next_node] ** alpha

                        # 检查距离是否为零，避免除零错误
                        distance = distances[current_node][next_node]
                        heuristic = (1.0 / distance) ** beta if distance != 0 else 1.0  # 替代为 1.0 或其他适当值

                        total_pheromone += pheromone * heuristic
                        probabilities.append((next_node, pheromone * heuristic))

                if total_pheromone == 0:
                    break

                # 轮盘赌选择下一个节点
                pick = random.uniform(0, total_pheromone)
                cumulative = 0.0
                for next_node, probability in probabilities:
                    cumulative += probability
                    if pick <= cumulative:
                        path.append(next_node)
                        visited[next_node] = True
                        current_node = next_node
                        break

            # 计算当前路径的总距离
            path_distance = calculate_total_distance(nodes, path)
            all_paths.append(path)
            all_distances.append(path_distance)

            # 更新全局最优解
            if path_distance < best_distance:
                best_path = path
                best_distance = path_distance

        # 信息素更新
        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1 - evaporation_rate)  # 信息素挥发

        # 所有蚂蚁完成后，在最佳路径上增加信息素
        for path, path_distance in zip(all_paths, all_distances):
            pheromone_contribution = pheromone_deposit / path_distance
            for i in range(len(path) - 1):
                pheromones[path[i]][path[i + 1]] += pheromone_contribution
                pheromones[path[i + 1]][path[i]] += pheromone_contribution

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
            path, total_distance = ant_colony_optimization(nodes)
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