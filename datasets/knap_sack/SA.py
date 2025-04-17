import random
import math
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

# 模拟退火算法（SA）求解 0-1 背包问题
def simulated_annealing(weights, values, capacity, num_iterations=1000, initial_temp=1000, cooling_rate=0.99):
    n = len(weights)

    # 初始解（随机选择物品）
    current_solution = np.random.randint(2, size=n)
    current_value = total_value(current_solution, values)
    current_weight = total_weight(current_solution, weights)

    # 初始温度
    temperature = initial_temp
    best_solution = current_solution.copy()
    best_value = current_value

    for iteration in range(num_iterations):
        # 在当前解的基础上产生一个邻域解（随机改变一个物品的选择）
        neighbor_solution = current_solution.copy()
        idx = random.randint(0, n-1)
        neighbor_solution[idx] = 1 - neighbor_solution[idx]  # 改变物品的选择（从0到1或从1到0）

        # 计算邻域解的价值和重量
        neighbor_value = total_value(neighbor_solution, values)
        neighbor_weight = total_weight(neighbor_solution, weights)

        # 如果邻域解有效且优于当前解，或者即使更差也按一定概率接受
        if neighbor_weight <= capacity and (neighbor_value > current_value or random.random() < math.exp((neighbor_value - current_value) / temperature)):
            current_solution = neighbor_solution
            current_value = neighbor_value
            current_weight = neighbor_weight

            # 更新最优解
            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value = current_value

        # 降低温度
        temperature *= cooling_rate

    return best_solution, best_value

# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            best_solution, best_value = simulated_annealing(weights, values, capacity)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"Best Solution (items selected): {best_solution}")
            print("-" * 50)

# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
