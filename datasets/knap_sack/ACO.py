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

# 蚁群算法求解 0-1 背包问题
def ant_colony_optimization(weights, values, capacity, num_ants=10, num_iterations=100, alpha=1, beta=1, evaporation_rate=0.1, pheromone_deposit=100):
    n = len(weights)
    pheromones = np.ones(n)  # 初始化信息素矩阵
    best_solution = None
    best_value = 0

    for iteration in range(num_iterations):
        solutions = []
        solution_values = []
        solution_weights = []

        for ant in range(num_ants):
            # 每个蚂蚁的解决方案：随机选择物品
            solution = [random.choice([0, 1]) for _ in range(n)]
            weight = total_weight(solution, weights)
            value = total_value(solution, values)

            # 评估解的有效性
            if weight <= capacity:
                solutions.append(solution)
                solution_values.append(value)
                solution_weights.append(weight)

                if value > best_value:
                    best_solution = solution
                    best_value = value

        # 更新信息素
        for i in range(n):
            pheromones[i] = (1 - evaporation_rate) * pheromones[i]  # 信息素挥发
            # 在选择最好的解中更新信息素
            for ant_solution in solutions:
                if ant_solution[i] == 1:
                    pheromones[i] += pheromone_deposit / total_value(ant_solution, values)  # 信息素增加

    return best_solution, best_value

# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            best_solution, best_value = ant_colony_optimization(weights, values, capacity)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"Best Solution (items selected): {best_solution}")
            print("-" * 50)

# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
