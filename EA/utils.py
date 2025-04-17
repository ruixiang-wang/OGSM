import os
import re
import json
from openai import OpenAI
from typing import List, Tuple

def parse_tsp_file(file_path: str) -> List[Tuple[int, int]]:
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

def generate_problem_description(cities: List[Tuple[int, int]]) -> str:
    description = "I have the following cities with their coordinates:\n"
    for i, (x, y) in enumerate(cities):
        description += f"City {i + 1}: (x: {x}, y: {y})\n"
    description += "Please solve the TSP (Traveling Salesman Problem) for these cities and provide the optimal path and total distance. If you have an answer, end your answer with total distance = your answer. If you don't have an answer, end your answer with total distance = 0."
    return description




def load_optimal_solutions(directory):
    optimal_solutions = {}
    optimal_solutions_file = os.path.join(directory, "optimal_solutions.txt")  # Correct the path

    print(f"Loading optimal solutions from: {optimal_solutions_file}")

    try:
        with open(optimal_solutions_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                match = re.match(r"([^:]+): \[.*\] \| Total Distance: (\d+)", line)
                if match:
                    tsp_file = match.group(1)
                    total_distance = float(match.group(2))
                    optimal_solutions[tsp_file] = total_distance
                else:
                    print(f"Skipping invalid line: {line}")
    except FileNotFoundError:
        print(f"Error: {optimal_solutions_file} not found.")
        return {}

    return optimal_solutions

def send_to_api(api_key: str, msg: str):
    client = OpenAI(api_key=api_key, base_url="https://api2.aigcbest.top/v1")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": msg}]
    )

    return response.choices[0].message.content


import json


def discuss_topic(problem: str, agents, rounds: int):
    discussion_history = []
    current_message = f"The topic for discussion is: {problem}. Let's begin focusing only on the solution to the problem."

    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1} ===")

        round_responses = {}

        for agent in agents:
            discussion_history.append({"agent_id": agent.agent_id, "message": current_message})

            response = agent.send_to_api(f"{agent.personality_prompt} {current_message} Please only focus on the solution.")
            # response = agent.send_to_api(f"{current_message} Please only focus on the solution.")

            analysis_message = "Based on the responses of the other agents, please reflect on your answer, revise it if necessary, and make sure your solution is optimal."
            response += f" {analysis_message}"

            round_responses[agent.agent_id] = response
            print(f"Agent {agent.agent_id} says: {response}")

        current_message = "\n".join([f"Agent {agent_id} says: {response}" for agent_id, response in round_responses.items()])

        for agent_id, response in round_responses.items():
            discussion_history.append({"agent_id": agent_id, "message": response})

    final_decision = "Based on the discussion, reflection, and analysis of other agents' answers, here is the final solution: "
    for agent in agents:
        decision_prompt = (
            f"The discussion topic was: {problem}. Here is the discussion history: {json.dumps(discussion_history)}. "
            "Please reflect on the entire discussion, summarize your refined solution, and provide a final conclusion based on the problem-solving and your analysis of other agents' answers."
        )
        final_decision += f"\nAgent {agent.agent_id}: {agent.send_to_api(agent.personality_prompt + decision_prompt)}"
        # final_decision += f"\nAgent {agent.agent_id}: {agent.send_to_api(decision_prompt)}"
    output = {
        "topic": problem,
        "discussion_history": discussion_history,
        "final_decision": final_decision,
    }
    with open("discussion_result.json", "w") as file:
        json.dump(output, file, indent=4)

    return final_decision

