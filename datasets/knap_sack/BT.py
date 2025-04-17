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

# 回溯法求解 0-1 背包问题
def backtracking(weights, values, capacity, n, solution, best_solution, best_value, current_weight, current_value, index):
    # 如果所有物品都被考虑过
    if index == n:
        if current_weight <= capacity and current_value > best_value:
            best_value = current_value
            best_solution = solution.copy()
        return best_solution, best_value

    # 选择当前物品
    solution[index] = 1
    new_weight = current_weight + weights[index]
    new_value = current_value + values[index]
    if new_weight <= capacity:
        best_solution, best_value = backtracking(weights, values, capacity, n, solution, best_solution, best_value, new_weight, new_value, index + 1)

    # 不选择当前物品
    solution[index] = 0
    best_solution, best_value = backtracking(weights, values, capacity, n, solution, best_solution, best_value, current_weight, current_value, index + 1)

    return best_solution, best_value

# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            n = len(weights)
            solution = [0] * n  # 当前物品选择情况
            best_solution = [0] * n  # 最优解
            best_value = 0
            best_solution, best_value = backtracking(weights, values, capacity, n, solution, best_solution, best_value, 0, 0, 0)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"Best Solution (items selected): {best_solution}")
            print("-" * 50)

# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
