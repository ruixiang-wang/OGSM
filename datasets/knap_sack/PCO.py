import random
import os
import numpy as np


# 读取数据文件
def read_knapsack_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        weights = list(map(int, lines[0].strip().split()))  # 物品的质量
        values = list(map(int, lines[1].strip().split()))  # 物品的价值
        capacity = int(lines[2].strip())  # 背包的容量
    return weights, values, capacity


# 计算背包的总价值
def total_value(solution, values):
    return sum(solution[i] * values[i] for i in range(len(solution)))


# 计算背包的总重量
def total_weight(solution, weights):
    return sum(solution[i] * weights[i] for i in range(len(solution)))


# 粒子群优化（PCO）求解 0-1 背包问题
def particle_swarm_optimization(weights, values, capacity, num_particles=10, num_iterations=100, inertia_weight=0.5,
                                cognitive_weight=1.5, social_weight=1.5):
    n = len(weights)

    # 初始化粒子的位置和速度
    particles = np.random.randint(2, size=(num_particles, n))  # 粒子的初始解（每个物品是否被选中）
    velocities = np.random.rand(num_particles, n)  # 粒子的速度
    personal_best_positions = particles.copy()  # 粒子的个人最佳解
    personal_best_values = np.array([total_value(p, values) for p in personal_best_positions])  # 个人最佳解的价值

    # 初始化全局最佳解
    global_best_position = personal_best_positions[np.argmax(personal_best_values)]
    global_best_value = max(personal_best_values)

    # 迭代更新粒子
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # 计算当前解的总重量，若超出容量，则禁用该解
            weight = total_weight(particles[i], weights)
            if weight > capacity:
                personal_best_values[i] = 0  # 无效解，设为零值
                continue

            # 计算适应度
            value = total_value(particles[i], values)

            # 更新个人最佳
            if value > personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = particles[i].copy()

            # 更新全局最佳
            if value > global_best_value:
                global_best_value = value
                global_best_position = particles[i].copy()

        # 更新粒子速度和位置
        for i in range(num_particles):
            for j in range(n):
                r1 = random.random()  # 随机数
                r2 = random.random()  # 随机数
                velocities[i][j] = (inertia_weight * velocities[i][j] +
                                    cognitive_weight * r1 * (personal_best_positions[i][j] - particles[i][j]) +
                                    social_weight * r2 * (global_best_position[j] - particles[i][j]))
                # 粒子的位置更新
                particles[i][j] = 1 if random.random() < 1 / (1 + np.exp(-velocities[i][j])) else 0

    return global_best_position, global_best_value


# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            best_solution, best_value = particle_swarm_optimization(weights, values, capacity)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"Best Solution (items selected): {best_solution}")
            print("-" * 50)


# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
