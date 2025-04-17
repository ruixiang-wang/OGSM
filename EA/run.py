import random
import os
from typing import List, Tuple
from agent import *
from EA import *
from prompt import *
from utils import *

def solve_and_select_best_agents(agents, directory, optimal_solutions):
    all_results = [[] for _ in range(len(agents))]
    tsp_answers_dict = {}

    for file_name in os.listdir(directory):
        if file_name.endswith(".tsp") and file_name.startswith("rue_15"):
            tsp_problem = "Solve the TSP problem."
            file_results = []
            for i, agent in enumerate(agents):
                tsp_answer = agent.solve_problem(tsp_problem, file_name)
                print(f"Result for {file_name} using Agent {agent.agent_id} with prompt '{agent.personality_prompt}':\n{tsp_answer}\n")
                total_distance = agent.extract_total_distance(tsp_answer)
                all_results[i].append(total_distance)

                if total_distance != 0:
                    agent_key = f"Agent{agent.agent_id} with prompt '{agent.personality_prompt}'"
                    file_results.append({agent_key: tsp_answer})

            if file_results:
                tsp_answers_dict[file_name] = file_results

    selected_agents, percent_differences = select_best_agents(all_results, optimal_solutions, agents)

    return selected_agents, percent_differences, all_results, tsp_answers_dict

def save_results_to_file(generation, selected_agents, percent_differences, all_results, output_file):
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(f"\n--- Generation {generation} ---\n")

        for idx, agent in enumerate(selected_agents):
            agent_id = agent.agent_id
            prompt = agent.personality_prompt
            agent_results = all_results[agent_id - 1]
            percent_diff = percent_differences[agent_id - 1]

            non_zero_values = [x for x in percent_diff if x != 0]
            if non_zero_values:
                avg_non_zero = sum(non_zero_values) / len(non_zero_values)
            else:
                avg_non_zero = 0.0

            file.write(f"Agent {agent_id} with prompt '{prompt}':\n")
            file.write(f"  Results: {agent_results}\n")
            file.write(f"  Percent Differences: {percent_diff}\n")
            file.write(f"  Average Percent Difference: {avg_non_zero:.6f}\n")

def save_selection_to_file(generation, response, select3_file):
    with open(select3_file, 'a', encoding='utf-8') as file:
        file.write(f"\n--- Generation {generation} ---\n")
        file.write(f"  Results: {response}\n")

def generate_tsp_prompt(tsp_answers_dict):
    prompt_lines = []

    for file_name, agents_answers in tsp_answers_dict.items():
        prompt_lines.append(f"For the file \"{file_name}\", here are the results for the agents:")

        for agent_answer in agents_answers:
            for agent_key, answer in agent_answer.items():
                prompt_lines.append(f"- {agent_key}: {answer}")

        prompt_lines.append("")

    return "\n".join(prompt_lines)

if __name__ == "__main__":
    api_key = "Your OpenAI key"
    agents = []

    # personality_prompts = [
    #     "",
    # ]
    #
    # for i, prompt in enumerate(personality_prompts, start=1):
    #     agent = LLM_Agent(personality_prompt=prompt, api_key=api_key, agent_id=i)
    #     agents.append(agent)

    ## MAS
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsp', 'create_problem', 'rue_5_1.tsp')
    cities = parse_tsp_file(file_path)
    problem_description = generate_problem_description(cities)
    final_decision = discuss_topic(problem_description, agents, 5)
    print(final_decision)

    ## read prompt from file for agent
    with open('prompts.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for index, line in enumerate(lines, 1):
            line = line.strip()
            agent = LLM_Agent(personality_prompt=line, api_key=api_key, agent_id=index)
            agents.append(agent)
    #
    # for agent in agents:
    #     print(f"Agent ID: {agent.agent_id}, Personality Prompt: {agent.personality_prompt}")

    # directory = os.path.join("..", "data", "knap_sack", "set")
    directory = os.path.join("..", "data", "tsp", "create_problem")
    # filename = "ks_20.txt"
    optimal_solutions_file = os.path.join("..", "data", "tsp")
    optimal_solutions = load_optimal_solutions(optimal_solutions_file)
    selected_agents = agents

    generations = 1
    output_file = "re_evo_ks_results_10.txt"
    # op = 172


    ### ks
    # for generation in range(generations):
    #     print(f"\n--- Generation {generation + 1}/{generations} ---")
    #
    #     response = crossover_mutation_llm(selected_agents, ea_prompts["crossover_mutation"], api_key)
    #     print(f"Generation {generation + 1} API Response: {response}")
    #
    #     if not response:
    #         print("No response received from API.")
    #         continue  # 跳过本次generation
    #
    #     response_lines = response.strip().split("\n")
    #     for line in response_lines:
    #         if line.startswith("Agent"):
    #             parts = line.split(":", 1)
    #             if len(parts) == 2:
    #                 agent_id = int(parts[0].split()[1])
    #                 new_prompt = parts[1].strip().strip('"')
    #                 selected_agents[agent_id - 1].personality_prompt = new_prompt
    #
    #     # 计算每个agent的final_value和difference，并记录在一个列表中
    #     agent_differences = []
    #     for agent in selected_agents:
    #         print(f"Agent ID: {agent.agent_id} Agent prompt: {agent.personality_prompt}")
    #         final_value = agent.extract_total_value(
    #             agent.send_to_api(agent.process_knapsack_data(directory, filename)))
    #         difference = abs(final_value - op) / op * 100  # 计算差异百分比
    #         agent_differences.append((agent.agent_id, agent.personality_prompt, final_value, difference))
    #
    #     if not agent_differences:
    #         print(f"No agent differences found for generation {generation + 1}")
    #         continue  # 如果没有任何agent差异，跳过该generation
    #
    #     # 根据difference值排序，选择difference最低的5个agent
    #     agent_differences.sort(key=lambda x: x[3])  # 按照difference升序排序
    #     selected_top_agents = agent_differences[:5]
    #
    #     # 检查是否选择了正确的top agents
    #     if not selected_top_agents:
    #         print(f"No top agents selected for generation {generation + 1}")
    #         continue
    #
    #     with open(output_file, "a") as f:
    #         # 将选中的5个agent的personality_prompt，final_value和difference记录到output_file
    #         f.write(f"--- Generation {generation + 1} ---\n")
    #         for agent_data in selected_top_agents:
    #             agent_id, prompt, final_value, difference = agent_data
    #             f.write(f"Agent ID: {agent_id}\n")
    #             f.write(f"Personality Prompt: {prompt}\n")
    #             f.write(f"Final Value: {final_value}\n")
    #             f.write(f"Difference: {difference:.2f}%\n")
    #             f.write("\n")
    #
    #     print(f"Generation {generation + 1} results have been written to {output_file}.")

    # select3_file = "select_3agents.txt"
    for generation in range(generations):
        print(f"\n--- Generation {generation + 1}/{generations} ---")

        response = crossover_mutation_llm(selected_agents, ea_prompts["crossover_mutation"], api_key)
        print(f"Generation {generation + 1} API Response: {response}")

        response_lines = response.strip().split("\n")
        for line in response_lines:
            if line.startswith("Agent"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    agent_id = int(parts[0].split()[1])
                    new_prompt = parts[1].strip().strip('"')
                    selected_agents[agent_id - 1].personality_prompt = new_prompt

        for i, agent in enumerate(selected_agents):
            print(f"Agent ID: {agent.agent_id} Agent prompt: {agent.personality_prompt}")


        selected_agents, percent_differences, all_results, tsp_answers_dict = solve_and_select_best_agents(selected_agents, directory, optimal_solutions)
        print(percent_differences)

        ## select
        # prompt_s = generate_tsp_prompt(tsp_answers_dict)
        #
        # additional_instructions = """
        # For all the TSP problem results below, consider the answers given by the agents across all the problems.
        # Based on their overall performance, select the top three agents who performed the best across all the problems.
        # For each of these selected agents, return their corresponding `personality_prompt`.
        # Now, considering all the answers for each agent across all the problems, choose the top 3 agents who performed the best overall. Return the `personality_prompt` for each of the selected agents.
        #
        # The most important thing: just return three sentence of agents' personality_prompt, don't return other information or sentence!!!
        # """
        # final_prompt = prompt_s + additional_instructions
        # select3_response = send_to_api(api_key, final_prompt)

        # print(select3_response)


        save_results_to_file(generation + 1, selected_agents, percent_differences, all_results, output_file)
        # save_selection_to_file(generation + 1,select3_response, select3_file)
