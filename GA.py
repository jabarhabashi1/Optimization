import random
import numpy as np
import matplotlib.pyplot as plt

def fitness_function(x):
    """
    Fitness function to evaluate the value of abs(4*x - 16).
    The goal is to minimize this value.
    """
    return abs(16 * x - 16)

def create_population(size, lower_bound, upper_bound):
    """
    Creates an initial population of random individuals.
    """
    return [random.uniform(lower_bound, upper_bound) for _ in range(size)]

def select_parents(population, objective_values):
    """
    Select two parents using a roulette wheel selection method.
    """
    total_fitness = sum(1 / (1 + val) for val in objective_values)
    probabilities = [(1 / (1 + val)) / total_fitness for val in objective_values]
    parents = np.random.choice(population, size=2, p=probabilities, replace=False)
    return parents

def crossover(parent1, parent2):
    """
    Perform single-point crossover between two parents.
    """
    alpha = random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def mutate(individual, mutation_rate, lower_bound, upper_bound):
    """
    Mutate an individual with a given mutation rate.
    """
    if random.random() < mutation_rate:
        individual += random.uniform(-1, 1)
        individual = max(min(individual, upper_bound), lower_bound)  # Keep within bounds
    return individual

def genetic_algorithm(
    fitness_function,
    population_size,
    generations,
    mutation_rate,
    lower_bound,
    upper_bound,
    tolerance,
    stopping_criteria
):
    """
    Genetic Algorithm to optimize the given fitness function.
    """
    # Initialize population
    population = create_population(population_size, lower_bound, upper_bound)
    fitness_history = []

    similar_count = 0
    last_best_fitness = None

    for generation in range(generations):
        # Evaluate objective values
        objective_values = [fitness_function(ind) for ind in population]

        # Create a new population
        new_population = []
        while len(new_population) < population_size:
            # Select parents
            parent1, parent2 = select_parents(population, objective_values)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutate
            child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
            child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

        # Find best individual
        best_individual = min(population, key=fitness_function)
        best_fitness = fitness_function(best_individual)
        fitness_history.append(best_fitness)

        # Check stopping criteria
        if last_best_fitness is not None and abs(last_best_fitness - best_fitness) < tolerance:
            similar_count += 1
        else:
            similar_count = 0
        last_best_fitness = best_fitness

        if similar_count >= stopping_criteria:
            print(f"Stopping early at generation {generation + 1}")
            break

        # Print best solution of the generation
        print(f"Generation {generation + 1}: Best solution = {best_individual}, Fitness = {best_fitness}")

    # Plot the fitness history
    plt.figure(figsize=(6, 4))
    plt.plot(fitness_history, label="Best Fitness (Objective Value)")
    plt.title("Convergence of Genetic Algorithm")
    plt.xlabel("Generations")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid()
    plt.show()

    # Return the best solution found
    return best_individual, best_fitness

# Parameters
population_size = 200
generations = 1000000
mutation_rate = 0.2  # Increased mutation rate
lower_bound = -10
upper_bound = 10
tolerance = 1e-2
stopping_criteria = 50

# Run the genetic algorithm
best_solution, best_fitness = genetic_algorithm(
    fitness_function,
    population_size,
    generations,
    mutation_rate,
    lower_bound,
    upper_bound,
    tolerance,
    stopping_criteria
)

print(f"\nOptimal solution found: x = {best_solution}, Objective value = {best_fitness}")
