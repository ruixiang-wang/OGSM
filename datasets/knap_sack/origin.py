import os
import random

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

# 最近邻算法求解 0-1 背包问题
def nearest_neighbor(weights, values, capacity):
    n = len(weights)
    solution = [0] * n  # 当前物品选择情况
    items = [(values[i] / weights[i] if weights[i] != 0 else 0, i) for i in range(n)]  # (价值/重量比, 物品索引)

    # 按照价值/重量比排序
    items.sort(reverse=True, key=lambda x: x[0])

    total_weight_value = 0  # 当前背包的总重量
    total_value_value = 0  # 当前背包的总价值

    for _, i in items:
        if total_weight_value + weights[i] <= capacity:  # 如果加上当前物品不超重
            solution[i] = 1  # 选择当前物品
            total_weight_value += weights[i]
            total_value_value += values[i]

    return solution, total_value_value

def nearest_insertion(weights, values, capacity):
    n = len(weights)
    solution = [0] * n  # 当前物品选择情况
    total_weight_value = 0  # 当前背包的总重量
    total_value_value = 0  # 当前背包的总价值

    # 初始化所有物品按价值排序，准备插入
    items = [(values[i], weights[i], i) for i in range(n)]  # (价值, 质量, 物品索引)
    items.sort(reverse=True, key=lambda x: x[0])  # 按照价值降序排序

    for value, weight, i in items:
        if total_weight_value + weight <= capacity:  # 如果加上当前物品不超重
            solution[i] = 1  # 选择当前物品
            total_weight_value += weight
            total_value_value += value

    return solution, total_value_value

def random_insertion(weights, values, capacity):
    n = len(weights)
    solution = [0] * n  # 当前物品选择情况
    total_weight_value = 0  # 当前背包的总重量
    total_value_value = 0  # 当前背包的总价值

    items = list(range(n))  # 物品索引
    random.shuffle(items)  # 随机打乱物品的顺序

    for i in items:
        if total_weight_value + weights[i] <= capacity:  # 如果加上当前物品不超重
            solution[i] = 1  # 选择当前物品
            total_weight_value += weights[i]
            total_value_value += values[i]

    return solution, total_value_value

def farthest_insertion(weights, values, capacity):
    n = len(weights)
    solution = [0] * n  # 当前物品选择情况
    total_weight_value = 0  # 当前背包的总重量
    total_value_value = 0  # 当前背包的总价值

    # 初始化：随机选择一个物品
    remaining_items = list(range(n))  # 所有物品的索引
    random.shuffle(remaining_items)  # 随机打乱物品顺序

    # 选择第一个物品加入背包
    first_item = remaining_items.pop()
    solution[first_item] = 1
    total_weight_value += weights[first_item]
    total_value_value += values[first_item]

    # 插入剩余的物品
    while remaining_items:
        # 计算每个剩余物品与当前背包物品的“最远距离”
        max_distance = -1
        insert_item = None
        for item in remaining_items:
            # 计算当前物品与背包中物品的“距离”
            # 这里的“距离”可以是物品之间的距离度量，比如计算物品的质量或价值差异
            distance = min(abs(weights[item] - weights[i]) for i in range(n) if solution[i] == 1)  # 最远插入

            if distance > max_distance:
                max_distance = distance
                insert_item = item

        # 插入最远物品
        remaining_items.remove(insert_item)
        solution[insert_item] = 1
        total_weight_value += weights[insert_item]
        total_value_value += values[insert_item]

    return solution, total_value_value

# 主程序：遍历数据集并求解背包问题
def solve_knapsack_problem(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            weights, values, capacity = read_knapsack_data(file_path)
            # solution, best_value = nearest_neighbor(weights, values, capacity)
            # solution, best_value = nearest_insertion(weights, values, capacity)
            # solution, best_value = random_insertion(weights, values, capacity)
            solution, best_value = farthest_insertion(weights, values, capacity)

            # 输出最优解
            print(f"Results for {filename}:")
            print(f"Best Value: {best_value}")
            print(f"Best Solution (items selected): {solution}")
            print("-" * 50)

# 调用主程序
if __name__ == "__main__":
    data_folder = './set'  # 数据集所在文件夹
    solve_knapsack_problem(data_folder)
