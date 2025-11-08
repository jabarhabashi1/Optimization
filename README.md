# Optimization Algorithms in Python 🚀🐍

This repository contains Python implementations of several **metaheuristic and black-box optimization algorithms**.  
Each implementation is kept simple and educational, with:

- A clear mathematical objective (test) function  
- A basic algorithmic core (population / swarm / sampling)  
- Optional early-stopping rules  
- A convergence plot (best value vs. iteration / trial)

You can use these scripts as **learning material**, as **templates for your own research**, or as **building blocks** inside larger projects.

---

## Algorithms Included

This repository currently implements:

- **Genetic Algorithm (GA)** 🧬 – evolution-inspired search using selection, crossover, and mutation  
- **Artificial Bee Colony (ABC)** 🐝 – swarm of bees exploring and exploiting food sources  
- **Particle Swarm Optimization (PSO)** 🐦 – particles flying through the search space guided by personal and global bests  
- **Grey Wolf Optimizer (GWO)** 🐺 – optimizer inspired by the social hierarchy and hunting behavior of grey wolves  
- **Ant Colony Optimization (ACO)** 🐜 – pheromone-based collective search, ideal for routing and combinatorial problems  
- **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** 🧠🔄 – powerful continuous optimizer with adaptive covariance  
- **Optuna-based Hyperparameter Optimization (Optuna Optimizer)** 🎯📊 – modern, sampler-based black-box optimizer with early stopping  

These algorithms effectively handle **non-linear**, **high-dimensional**, and **noisy** optimization problems where classical gradient-based methods struggle.

---

## 1. Genetic Algorithm (GA) 🧬

**Idea**  
Genetic Algorithm is inspired by **Darwinian evolution**. A population of candidate solutions evolves over generations using operators analogous to biology:

- **Selection** – better individuals are more likely to reproduce  
- **Crossover (recombination)** – combines information from two parents  
- **Mutation** – random changes that keep diversity in the population  

**Implemented in this repo**  
The GA script minimizes a 1D test function of the form:

> \( f(x) = \lvert a \cdot x - b \rvert \)

over a continuous interval. It uses:

- Roulette-wheel (fitness-proportional) selection  
- Blend / convex combination crossover  
- Bounded random mutation  
- Early stopping if the best fitness stops improving  

**Typical applications**

- 🎓 Feature selection and model structure search  
- 📅 Scheduling and timetabling  
- 🛠 Design and engineering optimization  
- 🧩 General global optimization where gradients are not available  

---

## 2. Artificial Bee Colony (ABC) 🐝

**Idea**  
ABC mimics the **foraging behavior of honeybee colonies**. Candidate solutions are modeled as food sources; bees explore and exploit these sources cooperatively:

- **Employed bees** explore around known sources and share information  
- **Onlooker bees** probabilistically choose promising sources to exploit  
- **Scout bees** abandon bad sources and randomly search new regions  

This balance between exploitation and exploration helps ABC escape local optima.

**Implemented in this repo**  

- Optimization of a simple continuous test function in 1D  
- Neighborhood search around current food sources  
- Abandonment and scout behavior to discover new areas  
- Tracking and plotting of best fitness vs. iteration  

**Typical applications**

- 🎯 Continuous function optimization  
- 🖼️ Image processing and segmentation  
- 📊 Clustering and data analysis  
- 📡 Wireless sensor network optimization  

---

## 3. Particle Swarm Optimization (PSO) 🐦

**Idea**  
PSO is inspired by the **collective motion of birds or fish**. Each particle is a candidate solution with:

- A **position** (current solution)  
- A **velocity** (direction and step size)  
- A **personal best** position  
- The swarm’s **global best** position  

Velocity updates blend:

- **Cognitive term** – “go towards my own best”  
- **Social term** – “go towards the swarm’s best”  
- Optional inertia to control momentum and exploration  

**Implemented in this repo**

- 1D PSO minimizing a simple objective \( f(x) = \lvert 4x - 16 \rvert \)  
- Bounded positions with velocity updates per iteration  
- Tracking of per-particle personal bests and global best  
- Early stopping based on lack of improvement  
- Convergence plot of global best vs. iteration  

**Typical applications**

- 🤖 Neural network training and parameter tuning  
- 🗓 Resource allocation and scheduling  
- 🔧 Continuous design and control problems  
- ⚙ Dynamic systems where the optimum may move over time  

---

## 4. Grey Wolf Optimizer (GWO) 🐺

**Idea**  
GWO is inspired by the **social hierarchy and cooperative hunting strategy** of grey wolves.  
Wolves are ranked into four main roles:

- **Alpha (α)** – leaders; the best solutions  
- **Beta (β)** – second-best; support and reinforce the alpha  
- **Delta (δ)** – third level; scouts, sentinels, and sub-leaders  
- **Omega (ω)** – the rest of the pack; exploration support  

The algorithm simulates:

1. **Encircling the prey** – wolves move around promising solutions  
2. **Hunting** – α, β, and δ guide the search  
3. **Attacking the prey** – convergence towards the best region  

By controlling position updates, GWO balances **exploration** and **exploitation**.

**Typical applications**

- ⚙ Engineering design optimization  
- ⭐ Feature selection and model tuning  
- 🔋 Energy management and renewable systems  
- 📐 Many real-valued continuous optimization tasks  

*(Implementation details in this repo follow the standard GWO logic; you can adapt the objective function to your own problem.)*

---

## 5. Ant Colony Optimization (ACO) 🐜

**Idea**  
ACO is based on how ants find the shortest paths to food using **pheromone trails**:

- Ants initially explore paths randomly  
- Better paths receive **higher pheromone concentration**  
- Pheromone evaporates over time, avoiding premature convergence  
- Over iterations, ants increasingly follow stronger pheromone paths  

This collective behavior makes ACO well-suited for **combinatorial and routing problems**.

**Implemented in this repo**

- A simplified ACO variant for 1D continuous minimization  
- Ants sample candidate positions and update a shared “pheromone” signal  
- New positions generated around promising areas (local search + exploration)  
- Early stopping based on small improvements in best fitness  
- Convergence curve plotting  

**Typical applications**

- 📦 Traveling Salesman Problem (TSP) and route planning  
- 📅 Job-shop scheduling and assignment problems  
- 🌐 Network routing and QoS optimization  

---

## 6. Covariance Matrix Adaptation Evolution Strategy (CMA-ES) 🧠🔄

**Idea**  
CMA-ES is a powerful **evolution strategy for continuous black-box optimization**.  
It maintains a **multivariate normal distribution** over the search space and:

- Samples a population of candidate points  
- Evaluates fitness and selects the best individuals  
- Updates the **mean** (search center)  
- Adapts the **covariance matrix** to learn promising directions  
- Adjusts the global step size (σ)  

This gives CMA-ES strong performance on **non-convex, ill-conditioned, and high-dimensional** problems.

**Implemented in this repo**

- Uses the `cma` Python library  
- Minimizes a simple multi-dimensional test function  
- Configurable dimension, bounds, and CMA-ES options  
- Tracks the best fitness per generation  
- Early stopping if improvement is below a threshold for several generations  
- Convergence plot of best fitness vs. generation  

**Typical applications**

- 🔬 Hyperparameter tuning for ML / DL models  
- 🤖 Robotics: controller and trajectory optimization  
- 🛰 Aerodynamic and structural engineering design  
- 📊 Financial modeling and trading strategy optimization  

---

## 7. Optuna-based Hyperparameter Optimization (Optuna Optimizer) 🎯📊

**Idea**  
Optuna is a **modern, Pythonic optimization framework** focused on hyperparameter tuning and general black-box optimization.  
Instead of hand-coding a metaheuristic, you:

1. Define an **objective function** that receives a `trial` (or parameter dict)  
2. Let Optuna’s **sampler** (e.g., TPE) propose parameters  
3. Optionally use **pruners / early-stopping** to cut off bad trials  

**Implemented in this repo (`Optuna_optimizer.py`)**

The script wraps Optuna in a generic function:

- You provide:
  - `objective_function(params: Dict[str, float]) -> float`  
  - `search_space: Dict[str, Tuple[float, float]]` for each real parameter  
- It uses a **TPE sampler** (optionally multivariate + grouping)  
- Two early-stopping callbacks:
  - **Plateau mode** – stop when there is no significant improvement for a set number of trials  
  - **Span mode** – stop when recent values are tightly clustered (almost flat)  
- A progress callback prints trial index, best value, and last value  
- At the end it:
  - Reports `best_params` and `best_value`  
  - Plots a convergence curve (best value so far vs. trial index)  

**Typical applications**

- 🔧 Hyperparameter tuning (learning rate, depth, regularization, etc.)  
- 📈 Calibrating parameters of trading or simulation models  
- 🧪 Any expensive, gradient-free objective where you just care about a scalar score  

Optuna complements the other metaheuristics by providing a **high-level, flexible framework** that can wrap any of your models or experiments.

---

## Advantages, Disadvantages, and Limitations ⚖️  

| Algorithm                                                            | Advantages                                                                                                                                                                                                                  | Disadvantages                                                                                                                               | Limitations                                                                                                                                            |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Genetic Algorithm (GA)** 🧬                                        | - Robust to local optima  <br> - Great for non-linear problems 🌐 <br> - Parallelizable 🖥️                                                                                                                                 | - Computationally expensive 💻  <br> - Slow convergence 🐌                                                                                  | Requires careful parameter tuning; not ideal for real-time tasks.                                                                                      |
| **Artificial Bee Colony (ABC)** 🐝                                   | - Simple 🛠️ <br> - Effective at global optima 🌎 <br> - Handles noisy functions well 🎵                                                                                                                                    | - Can stagnate on complex landscapes 🤔 <br> - Weaker performance in very high dimensions 🧮                                               | Best for continuous functions; struggles with purely discrete problems.                                                                                |
| **Particle Swarm Optimization (PSO)** 🐦                             | - Fast convergence ⚡ <br> - Few hyperparameters ✔️ <br> - Works well for dynamic systems 🔄                                                                                                                                | - Can get trapped in local optima 🚧 <br> - May require variants for multi-modal problems                                                  | Struggles with very rugged or discontinuous search spaces.                                                                                             |
| **Grey Wolf Optimizer (GWO)** 🐺                                     | - Balanced exploration & exploitation ⚖️ <br> - Minimal parameter tuning 🛠️                                                                                                                                                | - Risk of premature convergence ❌                                                                                                          | Limited theoretical guarantees; performance can degrade on highly complex, noisy real-world tasks.                                                     |
| **Ant Colony Optimization (ACO)** 🐜                                 | - Excellent for discrete and combinatorial problems 🧩 <br> - Scales to large instances 🏗️ <br> - Naturally adapts to changing environments 🔧                                                                            | - Computationally intensive 🖥️ <br> - Slow convergence ⏳                                                                                   | Primarily designed for combinatorial problems; continuous optimization requires additional modeling or hybridization.                                 |
| **CMA-ES** (Covariance Matrix Adaptation ES) 🧠🔄                    | - Highly adaptive search distribution 📊 <br> - Strong performance on continuous, high-dimensional problems 🔢 <br> - No gradients required                                         | - High computational cost per iteration 💰 <br> - Requires many function evaluations 🏃                                                    | Best suited for smooth, continuous spaces; less natural for discrete or heavily constrained domains.                                                  |
| **Optuna-based Hyperparameter Optimization (Optuna Optimizer)** 🎯📊 | - Flexible, model-agnostic framework 🧩 <br> - Built-in samplers (e.g., TPE) and pruners for early stopping ⏱️ <br> - Deep integration with Python ML ecosystem (PyTorch, TensorFlow, scikit-learn, etc.) 🤝               | - Adds a library dependency and orchestration overhead 📦 <br> - Quality depends heavily on good search spaces and objective definitions 🎯 | Best suited for black-box hyperparameter tuning; not a drop-in replacement for specialized metaheuristics in purely discrete or structured problems. |

---

## When to Use Which Algorithm? 🔍  

- Use **GA / ABC / PSO / GWO / CMA-ES** when you want **direct control over the metaheuristic** and possibly to customize its behavior.  
- Use **ACO** when your problem is **combinatorial / routing / graph-based**.  
- Use **Optuna** when you want a **high-level search framework** wrapped around your existing code, especially for ML hyperparameters.

💡 In practice, **hybrid approaches** (e.g., GA + local search, PSO + gradient steps, CMA-ES + Optuna orchestration) often give the best performance on challenging real-world problems.

---

## Getting Started 🧪

1. Clone or download the repository.  
2. Install required Python packages (e.g. `numpy`, `matplotlib`, `optuna`, `cma`):  

   ```bash
   pip install numpy matplotlib optuna cma
