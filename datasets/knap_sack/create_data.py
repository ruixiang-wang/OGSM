import random

def generate_knapsack_data(filename, num_items):
    weights = [random.randint(1, 20) for _ in range(num_items)]
    values = [random.randint(1, 20) for _ in range(num_items)]

    capacity = random.randint(20, num_items / 2 * 20)

    with open(filename, 'w') as file:
        file.write(" ".join(map(str, weights)) + "\n")
        file.write(" ".join(map(str, values)) + "\n")
        file.write(str(capacity) + "\n")


item_counts = [3, 5, 7, 10, 12, 15, 17, 20]

for count in item_counts:
    filename = f"ks_{count}.txt"
    generate_knapsack_data(filename, count)

print("Files generated successfully")
