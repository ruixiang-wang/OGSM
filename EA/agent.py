import os
from typing import List
from openai import OpenAI


class LLM_Agent:
    def __init__(self, personality_prompt: str, api_key: str, agent_id: int):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://api2.aigcbest.top/v1")
        self.personality_prompt = personality_prompt
        self.agent_id = agent_id  # Initialize agent with an ID

    def solve_problem(self, problem: str, tsp_file_name: str):
        print(f"Agent {self.agent_id} solving problem: {problem} using file {tsp_file_name}")
        if "TSP" in problem:
            return self.solve_tsp(tsp_file_name)
        return "Unknown problem"

    def solve_tsp(self, tsp_file_name: str):
        tsp_file_path = os.path.join("..", "data", "tsp", "create_problem", tsp_file_name)
        cities = self.parse_tsp_file(tsp_file_path)
        problem_description = self.generate_problem_description(cities)
        response = self.send_to_api(problem_description)
        return response

    def parse_tsp_file(self, file_path: str) -> List[tuple]:
        cities = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            start_reading = False
            for line in lines:
                line = line.strip()
                if not line: continue
                if line == "NODE_COORD_SECTION":
                    start_reading = True
                    continue
                if start_reading:
                    if line == "EOF":
                        break
                    parts = line.split()
                    if len(parts) < 3: continue
                    try:
                        city_id = int(parts[0])
                        x = int(parts[1])
                        y = int(parts[2])
                        cities.append((x, y))
                    except ValueError:
                        print(f"Invalid data on line: {line}")
                        continue
        return cities

    def generate_problem_description(self, cities: List[tuple]) -> str:
        description = f"{self.personality_prompt}\n\n"
        description += "I have the following cities with their coordinates:\n"
        for i, (x, y) in enumerate(cities):
            description += f"City {i + 1}: (x: {x}, y: {y})\n"
        description += "Please solve the TSP (Traveling Salesman Problem) for these cities and provide the optimal path and total distance. If you have an answer, end your answer with total distance = your answer. If you don't have an answer, end your answer with total distance = 0."
        return description

    def process_knapsack_data(self, folder_path: str, filename: str) -> str:
        file_path = os.path.join(folder_path, filename)

        items = []
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                weights = list(map(int, lines[0].strip().split()))
                values = list(map(int, lines[1].strip().split()))
                capacity = int(lines[2].strip())

                for i in range(len(weights)):
                    items.append((f"item{i + 1}", values[i], weights[i]))

        except Exception as e:
            return f"Error reading file: {str(e)}"

        prompt = f"{self.personality_prompt} Here is a 0/1 knapsack problem: \n\n"
        prompt += "The problem involves the following items:\n"

        for item in items:
            name, value, weight = item
            prompt += f"- {name}: Value = {value}, Weight = {weight}\n"

        prompt += f"\nThe capacity of the knapsack is {capacity}.\n"
        prompt += "Please solve this 0/1 knapsack problem. Maximize the total value without exceeding the given weight limit.If you have an answer, end your answer with total value = your answer. If you don't have an answer, end your answer with total value = 0."

        return prompt

    def send_to_api(self, msg: str):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": msg}]
        )
        return response.choices[0].message.content

    def extract_total_distance(self, response: str) -> float:
        prompt = f"Please extract the total distance from the following response: {response}\nTotal distance:"
        api_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        extracted_response = api_response.choices[0].message.content.strip()
        try:
            total_distance = float(extracted_response)
            return total_distance
        except ValueError:
            return 0.0

    def extract_total_value(self, response: str) -> float:
        prompt = f"Please extract the total value from the following response: {response}\nTotal value:"
        api_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        extracted_response = api_response.choices[0].message.content.strip()
        try:
            total_distance = float(extracted_response)
            return total_distance
        except ValueError:
            return 0.0


