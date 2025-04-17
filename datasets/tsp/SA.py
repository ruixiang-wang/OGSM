import os
import math
import random

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def read_tsp_file(filename):
    nodes = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == "NODE_COORD_SECTION":
                break
        for line in lines[lines.index("NODE_COORD_SECTION\n") + 1:]:
            parts = line.split()
            if len(parts) == 3:
                nodes.append((float(parts[1]), float(parts[2])))
    return nodes


def calculate_total_distance(nodes, path):
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += euclidean_distance(nodes[path[i]], nodes[path[i + 1]])
    total_distance += euclidean_distance(nodes[path[-1]], nodes[path[0]])  # 回到起点
    return total_distance


def generate_neighbor(path):
    new_path = path[:]
    i, j = random.sample(range(len(path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def simulated_annealing(nodes, initial_temp=1000, cooling_rate=0.995, max_iter=10000):
    n = len(nodes)
    current_path = list(range(n))
    random.shuffle(current_path)
    current_distance = calculate_total_distance(nodes, current_path)

    best_path = current_path[:]
    best_distance = current_distance

    temperature = initial_temp
    iteration = 0

    while temperature > 1e-3 and iteration < max_iter:
        new_path = generate_neighbor(current_path)
        new_distance = calculate_total_distance(nodes, new_path)

        if new_distance < current_distance:
            current_path = new_path
            current_distance = new_distance
            if current_distance < best_distance:
                best_path = current_path[:]
                best_distance = current_distance
        else:
            acceptance_prob = math.exp((current_distance - new_distance) / temperature)
            if random.random() < acceptance_prob:
                current_path = new_path
                current_distance = new_distance

        temperature *= cooling_rate
        iteration += 1

    return best_path, best_distance


def output_result(filename, path, total_distance, optimal_distance):
    print(f"File: {filename} - Shortest Path: {path} - Total Distance: {total_distance:.6f} - Difference: {abs(total_distance - optimal_distance) / optimal_distance * 100:.2f}%")


def read_optimal_solutions(filename):
    optimal_solutions = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_name = line.split(":")[0].strip()  # 去掉 .tsp 后缀
            total_distance = float(line.split("Total Distance: ")[1].strip())
            optimal_solutions[file_name] = total_distance
    return optimal_solutions

def solve_all_tsp_in_directory(directory, optimal_solutions):
    for filename in os.listdir(directory):
        if filename.endswith('.tsp'):
            filepath = os.path.join(directory, filename)
            nodes = read_tsp_file(filepath)
            path, total_distance = simulated_annealing(nodes)
            if filename in optimal_solutions:
                optimal_distance = optimal_solutions[filename]
                output_result(filename, path, total_distance, optimal_distance)
            else:
                print(f"Optimal solution for {filename} not found.")


# 主函数
if __name__ == "__main__":
    directory = './create_problem'
    optimal_solutions = read_optimal_solutions('optimal_solutions.txt')
    solve_all_tsp_in_directory(directory, optimal_solutions)
