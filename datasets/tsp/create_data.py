import random
import os

def generate_tsp_data(node_counts, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for count in node_counts:
        for i in range(1, 11):  # 10 datasets for each node count
            # Generate a name based on the format "rue_<node_count>_<dataset_number>.tsp"
            filename = os.path.join(folder_name, f"rue_{count}_{i}.tsp")

            # Generate random coordinates for the nodes
            nodes = []
            for node in range(1, count + 1):
                x = random.randint(1, 40)
                y = random.randint(1, 40)
                nodes.append(f"{node} {x} {y}")

            # Write the TSP formatted output to the file
            with open(filename, 'w') as f:
                f.write(f"NAME: rue_{count}_{i}\n")
                f.write("TYPE: TSP\n")
                f.write(f"DIMENSION: {count}\n")
                f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
                f.write("NODE_COORD_SECTION\n")
                f.write("\n".join(nodes) + "\n")
                f.write("\n")  # Print a blank line to separate datasets


# Generate data for 5, 10, 15, 20, 25, and 30 nodes and save each dataset as a .tsp file in 'create_problem' folder
generate_tsp_data([5, 10, 15, 20, 25, 30], 'create_problem')
