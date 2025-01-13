import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_particles = 10      # Number of particles
n_iterations = 10000    # Maximum number of iterations
c1 = 2.0              # Cognitive parameter
c2 = 2.0              # Social parameter
omega = 0.5           # Inertia weight
x_min, x_max = 0, 10  # Search space boundaries
v_max = 1.0           # Maximum velocity
tolerance = 1e-3      # Tolerance for stopping
stopping_criteria = 10 # Stopping threshold for similar results

# Objective function
def objective_function(x):
    return abs(4 * x - 16)

# Initialize particles
positions = np.random.uniform(x_min, x_max, n_particles)
velocities = np.random.uniform(-v_max, v_max, n_particles)
best_personal_positions = np.copy(positions)
best_personal_scores = np.array([objective_function(x) for x in positions])
global_best_position = positions[np.argmin(best_personal_scores)]
global_best_score = min(best_personal_scores)

# Tracking variables
fitness_history = []
similar_count = 0
last_best_score = None

# PSO main loop
for iteration in range(n_iterations):
    for i in range(n_particles):
        # Update velocity
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (
            omega * velocities[i]
            + c1 * r1 * (best_personal_positions[i] - positions[i])
            + c2 * r2 * (global_best_position - positions[i])
        )
        # Limit velocity
        velocities[i] = np.clip(velocities[i], -v_max, v_max)

        # Update position
        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], x_min, x_max)

        # Evaluate objective function
        score = objective_function(positions[i])

        # Update personal best
        if score < best_personal_scores[i]:
            best_personal_scores[i] = score
            best_personal_positions[i] = positions[i]

    # Update global best
    current_global_best = np.argmin(best_personal_scores)
    if best_personal_scores[current_global_best] < global_best_score:
        global_best_score = best_personal_scores[current_global_best]
        global_best_position = best_personal_positions[current_global_best]

    # Track fitness history
    fitness_history.append(global_best_score)

    # Check stopping criteria
    if last_best_score is not None and abs(last_best_score - global_best_score) < tolerance:
        similar_count += 1
    else:
        similar_count = 0
    last_best_score = global_best_score

    if similar_count >= stopping_criteria:
        print(f"Stopping early at iteration {iteration + 1}")
        break

# Print final result
print(f"Optimal solution found at x = {global_best_position:.5f} with f(x) = {global_best_score:.5f}")

# Plot the fitness history
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fitness_history) + 1), fitness_history)
plt.title("Fitness over Iterations in PSO")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
