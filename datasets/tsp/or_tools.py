import os
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# 计算两点之间的欧几里得距离
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 读取tsp文件并解析出坐标数据
def read_tsp_file(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        reading_coordinates = False
        for line in lines:
            line = line.strip()  # 去掉每行的多余空格和换行符
            if line.startswith("NODE_COORD_SECTION"):
                reading_coordinates = True
                continue
            if reading_coordinates:
                if line.strip() == 'EOF':  # 文件结束标志
                    break
                if not line:  # 跳过空行
                    continue
                parts = line.split()
                if len(parts) == 3:  # 确保每行有3个元素 (城市编号、x坐标、y坐标)
                    try:
                        # 解析坐标，并添加到列表中
                        coordinates.append((float(parts[1]), float(parts[2])))  # 只提取坐标
                    except ValueError:
                        print(f"Error parsing line: {line}")  # 如果转换失败，打印出错行
                else:
                    print(f"Skipping invalid line: {line}")  # 如果格式不对，跳过该行

    print(f"Coordinates from {file_path}: {coordinates}")  # 调试输出坐标
    return coordinates


# 创建距离矩阵
def create_data_model(coordinates):
    data = {}
    n = len(coordinates)
    data['distance_matrix'] = [
        [euclidean_distance(coordinates[i], coordinates[j]) for j in range(n)]
        for i in range(n)
    ]
    data['num_vehicles'] = 1  # 只有一辆车
    data['depot'] = 0  # 起点
    return data


# 使用 OR-Tools 求解 TSP 问题
def solve_tsp(coordinates):
    # 数据准备
    data = create_data_model(coordinates)

    # 检查数据模型
    print(f"Distance matrix: {data['distance_matrix']}")

    # 创建路由模型
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # 创建距离回调函数
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 设置成本函数
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 搜索参数设置
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # 求解
    solution = routing.SolveWithParameters(search_parameters)

    # 输出结果
    if solution:
        total_distance = 0
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        return route, total_distance
    else:
        return None, None


# 主程序
def main():
    tsp_folder = './create_problem'
    output_file = 'optimal_solutions.txt'

    with open(output_file, 'w') as result_file:
        for filename in os.listdir(tsp_folder):
            if filename.endswith('.tsp'):
                file_path = os.path.join(tsp_folder, filename)

                # 读取 TSP 文件
                coordinates = read_tsp_file(file_path)

                # 求解 TSP 问题
                route, total_distance = solve_tsp(coordinates)

                if route is not None:
                    result_file.write(f"{filename}: {route} | Total Distance: {total_distance}\n")
                else:
                    result_file.write(f"{filename}: No solution found\n")

    print(f"Results have been saved to {output_file}")


if __name__ == '__main__':
    main()
