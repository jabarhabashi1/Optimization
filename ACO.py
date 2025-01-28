import numpy as np
import matplotlib.pyplot as plt

# Objective function  
def fitness_function(x):
    return abs(4 * x - 16)

def aco_minimization_with_plot(fitness_function, n_ants, max_iter, lower_bound, upper_bound,
                               evaporation_rate=0.5, alpha=1, beta=2, tolerance=1e-6, stopping_criteria=60):
    # Initialize pheromones and ants
    pheromone = np.ones(n_ants)  # Initial pheromone concentration
    ants = np.random.uniform(lower_bound, upper_bound, n_ants)
    best_position = ants[0]
    best_fitness = float('inf')

    fitness_history = []
    similar_count = 0
    last_best_fitness = None

    for iteration in range(max_iter):
        # Calculate fitness and inverse fitness (higher fitness = lower quality)
        fitness_values = np.array([fitness_function(x) for x in ants])
        inverse_fitness = 1 / (fitness_values + 1e-10)  # Avoid division by zero

        # Update pheromone with evaporation and reinforcement
        pheromone = (1 - evaporation_rate) * pheromone + alpha * inverse_fitness

        # Calculate selection probabilities
        probabilities = pheromone ** beta * inverse_fitness ** alpha
        probabilities /= probabilities.sum()  # Normalize

        # Generate new positions using Gaussian exploration around selected ants
        new_ants = []
        for _ in range(n_ants):
            selected = np.random.choice(ants, p=probabilities)
            new_pos = selected + np.random.normal(0, 0.5)  # Local search
            new_pos = np.clip(new_pos, lower_bound, upper_bound)
            new_ants.append(new_pos)
        ants = np.array(new_ants)

        # Update best solution
        current_best_idx = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_position = ants[current_best_idx]

        fitness_history.append(best_fitness)

        # Stopping criteria (no improvement for N iterations)
        if last_best_fitness is not None and abs(last_best_fitness - best_fitness) < tolerance:
            similar_count += 1
        else:
            similar_count = 0
        last_best_fitness = best_fitness

        if similar_count >= stopping_criteria:
            print(f"Early stopping at iteration {iteration + 1}")
            break

    return best_position, best_fitness, fitness_history


# Parameters
n_ants = 5
max_iter = 1000000
lower_bound = -10
upper_bound = 10
tolerance = 1e-8
stopping_criteria = 50

# Run ACO
best_position, best_fitness, fitness_history = aco_minimization_with_plot(
    fitness_function, n_ants, max_iter, lower_bound, upper_bound,
    evaporation_rate=0.5, alpha=1, beta=2, tolerance=tolerance, stopping_criteria=stopping_criteria
)

# Results
print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fitness_history) + 1), fitness_history, color='red')
plt.title("Fitness Convergence in ACO")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()