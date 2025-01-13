import numpy as np
import matplotlib.pyplot as plt

# Objective function
def fitness_function(x):
    return abs(4 * x - 16)

def gwo_minimization_with_plot(fitness_function, n_wolves, max_iter, lower_bound, upper_bound, tolerance=1e-6, stopping_criteria=60):
    wolves = np.random.uniform(lower_bound, upper_bound, n_wolves)
    Alpha, Beta, Delta = wolves[0], wolves[1], wolves[2]  # Initialize Alpha, Beta, Delta to valid positions
    Alpha_fitness, Beta_fitness, Delta_fitness = float('inf'), float('inf'), float('inf')

    for wolf in wolves:
        fitness = fitness_function(wolf)
        if fitness < Alpha_fitness:
            Alpha, Alpha_fitness = wolf, fitness
        elif fitness < Beta_fitness:
            Beta, Beta_fitness = wolf, fitness
        elif fitness < Delta_fitness:
            Delta, Delta_fitness = wolf, fitness

    fitness_history = []
    similar_count = 0
    last_best_fitness = None

    for iteration in range(max_iter):
        for i in range(n_wolves):
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2 * r1 - 1, 2 * r2
            D_alpha = abs(C1 * Alpha - wolves[i])
            X1 = Alpha - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2, C2 = 2 * r1 - 1, 2 * r2
            D_beta = abs(C2 * Beta - wolves[i])
            X2 = Beta - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3, C3 = 2 * r1 - 1, 2 * r2
            D_delta = abs(C3 * Delta - wolves[i])
            X3 = Delta - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3
            wolves[i] = np.clip(wolves[i], lower_bound, upper_bound)

        for wolf in wolves:
            fitness = fitness_function(wolf)
            if fitness < Alpha_fitness:
                Alpha, Alpha_fitness = wolf, fitness
            elif fitness < Beta_fitness:
                Beta, Beta_fitness = wolf, fitness
            elif fitness < Delta_fitness:
                Delta, Delta_fitness = wolf, fitness

        fitness_history.append(Alpha_fitness)

        # Check stopping criteria
        if last_best_fitness is not None and abs(last_best_fitness - Alpha_fitness) < tolerance:
            similar_count += 1
        else:
            similar_count = 0
        last_best_fitness = Alpha_fitness

        if similar_count >= stopping_criteria:
            print(f"Stopping early at iteration {iteration + 1}")
            break

    return Alpha, Alpha_fitness, fitness_history

# Parameters
n_wolves = 5
max_iter = 1000000
lower_bound = -10
upper_bound = 10
tolerance = 1e-2
stopping_criteria = 50

# Run GWO with fitness history tracking
best_position, best_fitness, fitness_history = gwo_minimization_with_plot(
    fitness_function, n_wolves, max_iter, lower_bound, upper_bound, tolerance, stopping_criteria
)

# Print final result
print(f"Best Position: {best_position}, Best Fitness: {best_fitness}")

# Plot fitness over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fitness_history) + 1), fitness_history)
plt.title("Fitness over Iterations in GWO")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
