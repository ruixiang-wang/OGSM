import os
import math
import random
import numpy as np

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

# 粒子群优化（PCO）算法求解TSP问题
def particle_swarm_optimization(nodes, num_particles=10, num_iterations=100, w=0.5, c1=1.5, c2=1.5):
    n = len(nodes)
    distances = [[euclidean_distance(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]

    # 初始化粒子的位置和速度
    particles = []
    for _ in range(num_particles):
        path = list(range(n))
        random.shuffle(path)
        velocity = [random.randint(-1, 1) for _ in range(n)]
        particles.append({
            'position': path,
            'velocity': velocity,
            'best_position': path.copy(),
            'best_distance': float('inf')
        })

    global_best_position = None
    global_best_distance = float('inf')

    # 粒子群优化主循环
    for iteration in range(num_iterations):
        for particle in particles:
            # 计算当前粒子的路径距离
            path_distance = calculate_total_distance(nodes, particle['position'])

            # 更新粒子的个人最优解
            if path_distance < particle['best_distance']:
                particle['best_distance'] = path_distance
                particle['best_position'] = particle['position'].copy()

            # 更新全局最优解
            if path_distance < global_best_distance:
                global_best_distance = path_distance
                global_best_position = particle['position'].copy()

        # 更新粒子的速度和位置
        for particle in particles:
            new_velocity = []
            new_position = particle['position'][:]

            for i in range(n):
                # 更新速度：惯性 + 个人经验 + 全局经验
                inertia = w * particle['velocity'][i]
                cognitive = c1 * random.random() * (particle['best_position'][i] - particle['position'][i])
                social = c2 * random.random() * (global_best_position[i] - particle['position'][i])
                new_velocity.append(inertia + cognitive + social)

                # 更新位置：基于速度调整路径
                new_position[i] = int(new_position[i] + new_velocity[i]) % n

            # 更新粒子的位置和速度
            particle['velocity'] = new_velocity
            particle['position'] = new_position

        print(f"Iteration {iteration + 1}/{num_iterations}: Global Best Distance: {global_best_distance:.2f}")

    return global_best_position, global_best_distance

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
            path, total_distance = particle_swarm_optimization(nodes)
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
