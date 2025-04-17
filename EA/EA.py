from typing import List
from utils import *

def select_best_agents(all_results, optimal_solutions, agents):
    agent_scores = []
    percent_differences = []

    for i, agent_results in enumerate(all_results):
        agent_percent_diff = []
        score = 0
        for j, result in enumerate(agent_results):
            if j == 0:
                tsp_file = "rue_5_1.tsp"
            elif j == 1:
                tsp_file = "rue_5_10.tsp"
            else:
                tsp_file = f"rue_5_{j}.tsp"
            optimal_solution = optimal_solutions.get(tsp_file)

            if optimal_solution is not None and result != 0:
                score += 1 / (abs(result - optimal_solution) + 1)

                percent_diff = abs(result - optimal_solution) / optimal_solution * 100
            else:
                percent_diff = 0

            agent_percent_diff.append(percent_diff)

        percent_differences.append(agent_percent_diff)
        agent_scores.append((i, score))

    agent_scores = sorted(agent_scores, key=lambda x: x[1], reverse=True)

    selected_agents = [agents[i] for i, _ in agent_scores[:7]]

    return selected_agents, percent_differences

def crossover_mutation_llm(selected_agents: List[str], ea_prompts: str, api_key: str) -> str:
    formatted_agents = "\n".join([f"{i + 1}. {agent}" for i, agent in enumerate(selected_agents)])

    # final_prompt = ea_prompts.replace("### Final Notes:", f"## Step 1: Analyze the Input Prompts\nThe following 7 agents are selected:\n{formatted_agents}\n### Final Notes:")

    response = send_to_api(api_key, ea_prompts)
    return response
