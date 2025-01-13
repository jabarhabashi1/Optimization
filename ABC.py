import matplotlib.pyplot as plt
import numpy as np
# Initialize parameters
max_iterations = 10000  # A large number to simulate "infinite" iterations
tolerance = 1e-6  # Threshold for similarity
stopping_criteria = 60  # Number of similar results to stop
n = 20   # Total bees
m = 10    # Food sources
e = 4    # Elite sources
ngh = 0.5  # Neighborhood radius
n1 = 1   # Bees for elite sources
n2 = 3   # Bees for other sources
x_min, x_max = 0, 5  # Search range
# Objective function
def objective_function(x):
    return np.abs(14.5 * x - 16)

# Tracking variables
fitness_history = []

# Reinitialize food sources
food_sources = np.random.uniform(x_min, x_max, m)
fitness = objective_function(food_sources)

# ABC algorithm with stopping condition
similar_count = 0
last_best_fitness = None

for iteration in range(max_iterations):
    # Sort food sources by fitness (ascending order)
    sorted_indices = np.argsort(fitness)
    food_sources = food_sources[sorted_indices]
    fitness = fitness[sorted_indices]

    # Generate new solutions for elite food sources
    for i in range(e):
        for _ in range(n1):
            new_solution = food_sources[i] + ngh * np.random.randn()
            new_solution = np.clip(new_solution, x_min, x_max)  # Keep within bounds
            new_fitness = objective_function(new_solution)
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness

    # Generate new solutions for non-elite food sources
    for i in range(e, m):
        for _ in range(n2):
            new_solution = food_sources[i] + ngh * np.random.randn()
            new_solution = np.clip(new_solution, x_min, x_max)  # Keep within bounds
            new_fitness = objective_function(new_solution)
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness

    # Scout bees: replace worst sources with random solutions
    for i in range(e, m):
        if np.random.rand() < 0.1:  # 10% chance of replacement
            food_sources[i] = np.random.uniform(x_min, x_max)
            fitness[i] = objective_function(food_sources[i])

    # Track best fitness
    best_index = np.argmin(fitness)
    best_fitness = fitness[best_index]
    fitness_history.append(best_fitness)

    # Check stopping criteria
    if last_best_fitness is not None and abs(last_best_fitness - best_fitness) < tolerance:
        similar_count += 1
    else:
        similar_count = 0
    last_best_fitness = best_fitness

    if similar_count >= stopping_criteria:
        break
# Final best solution
best_solution = food_sources[best_index]
best_fitness = fitness[best_index]

print('Iteration:',iteration, 'Best_solution:', best_solution, 'Best_fitness:', best_fitness)
# Plot the fitness history
plt.figure(figsize=(6, 4))
plt.plot(fitness_history, label="Best Fitness")
plt.title("Convergence of ABC Algorithm")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.legend()
plt.grid()
plt.show()

