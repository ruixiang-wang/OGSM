import re
import os
import shutil

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.lstrip("●. ")
            match = re.match(r'([^:]+:\d+)', line)
            if match:
                cleaned_line = match.group(1)
                outfile.write(cleaned_line + '\n')

def organize_tsp_and_tour_files():
    base_directory = os.getcwd()
    all_tsp_folder = os.path.join(base_directory, 'ALL_tsp')
    problem_folder = os.path.join(base_directory, 'problem')
    solution_folder = os.path.join(base_directory, 'solution')

    if not os.path.exists(problem_folder):
        os.makedirs(problem_folder)

    if not os.path.exists(solution_folder):
        os.makedirs(solution_folder)

    for filename in os.listdir(all_tsp_folder):
        file_path = os.path.join(all_tsp_folder, filename)

        if os.path.isfile(file_path):
            if filename.endswith('.tsp'):
                shutil.move(file_path, problem_folder)
            elif filename.endswith('.tour'):
                shutil.move(file_path, solution_folder)

if __name__ == "__main__":
    print(" ")
    # organize_tsp_and_tour_files()
    # input_file = 'input.txt'
    # output_file = 'opt.txt'
    # process_file(input_file, output_file)

