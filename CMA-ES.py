""" Covariance matrix adaptation evolution strategy (CMA-ES) """
import cma
import numpy as np
import matplotlib.pyplot as plt


def fitness_function(x):
    """Fitness function to minimize the value of abs(16x - 16) for any dimension."""
    return np.sum(np.abs(16 * x - 16))  # Sum of errors across all dimensions


# Algorithm parameters
n_dimensions = 1  # Number of dimensions (e.g., 1 for 1D problem)
initial_mean = np.zeros(n_dimensions)  # Initial starting point in each dimension
initial_sigma = 1  # Initial step size
population_size = 200  # Population size
lower_bound = -10  # Lower bound
upper_bound = 10  # Upper bound
max_generations = 1000000  # Maximum number of generations
tolerance = 1e-5  # Tolerance threshold for stopping
stopping_criteria = 500  # Number of generations without improvement for stopping

# Configure CMA-ES options
opts = cma.CMAOptions()
opts.set('popsize', population_size)
opts.set('bounds', [lower_bound, upper_bound])
opts.set('maxiter', max_generations)

# Initialize CMA-ES
es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, opts)

best_fitness_history = []  # History of best fitness values
similar_count = 0  # Counter for generations without improvement
last_best_fitness = None  # Best fitness of the previous generation
overall_best_solution = None  # Overall best solution
overall_best_fitness = float('inf')  # Overall best fitness

# Run the algorithm
for generation in range(max_generations):
    # Generate new population
    solutions = es.ask()

    # Evaluate fitness of the population
    fitness_values = [fitness_function(x) for x in solutions]

    # Update CMA-ES with fitness values
    es.tell(solutions, fitness_values)

    # Find the best solution and fitness in the current generation
    current_best_idx = np.argmin(fitness_values)
    current_best_fitness = fitness_values[current_best_idx]
    current_best_solution = solutions[current_best_idx]

    # Update the overall best solution
    if current_best_fitness < overall_best_fitness:
        overall_best_fitness = current_best_fitness
        overall_best_solution = current_best_solution

    # Store fitness history
    best_fitness_history.append(current_best_fitness)

    # Check stopping condition
    if last_best_fitness is not None:
        if abs(last_best_fitness - current_best_fitness) < tolerance:
            similar_count += 1
        else:
            similar_count = 0
    last_best_fitness = current_best_fitness

    if similar_count >= stopping_criteria:
        print(f"\nEarly stopping at generation {generation + 1}")
        break

    # Display progress
    print(f"Generation {generation + 1}: Best fitness = {current_best_fitness:.4f}", end='\r')

# Plot convergence
plt.figure(figsize=(6, 4))
plt.plot(best_fitness_history, label='Best Fitness')
plt.title('Convergence of CMA-ES Algorithm')
plt.xlabel('Generations')
plt.ylabel('Objective Value')
plt.legend()
plt.grid()
plt.show()

print(f"\nBest solution found: x = {overall_best_solution}")
print(f"Objective value: {overall_best_fitness:.6f}")