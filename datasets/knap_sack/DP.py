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


# 动态规划算法（DP）求解 0-1 背包问题
def dynamic_programming(weights, values, capacity):
    n = len(weights)

    # dp[i][w] 表示前 i 个物品，在背包容量为 w 时的最大价值
    dp = np.zeros((n + 1, capacity + 1), dtype=int)

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # 最终解在 dp[n][capacity] 中
    return dp[n][capacity], dp


# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            best_value, dp_table = dynamic_programming(weights, values, capacity)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"DP Table (last row): {dp_table[-1]}")
            print("-" * 50)


# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
