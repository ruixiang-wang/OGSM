import random
import numpy as np
import os


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


# 贪心算法求解 0-1 背包问题
def greedy_algorithm(weights, values, capacity):
    n = len(weights)

    # 计算每个物品的价值密度 (value/weight)
    value_density = [(values[i] / weights[i], i) for i in range(n)]

    # 按照价值密度从大到小排序
    value_density.sort(reverse=True, key=lambda x: x[0])

    total_value = 0
    total_weight = 0
    selected_items = [0] * n  # 用于记录每个物品是否被选中

    for density, i in value_density:
        if total_weight + weights[i] <= capacity:  # 如果当前物品可以放入背包
            selected_items[i] = 1
            total_value += values[i]
            total_weight += weights[i]

    return selected_items, total_value


# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            best_solution, best_value = greedy_algorithm(weights, values, capacity)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"Best Solution (items selected): {best_solution}")
            print("-" * 50)


# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
