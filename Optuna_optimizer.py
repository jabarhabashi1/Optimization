import numpy as np
import optuna
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional

# ---------- Early stopping callbacks (extracted and simplified) ----------


def make_plateau_cb(
    min_trials: int = 50,
    patience: int = 20,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
):
    """
    If, after at least `min_trials` have been completed, there is no
    significant improvement (greater than the threshold) within the last
    `patience` trials, the study is stopped.
    This is written for the `minimize` direction.
    """

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed = [
            t for t in study.trials
            if (t.value is not None and t.state.name == "COMPLETE")
        ]
        if len(completed) < (min_trials + patience):
            return

        # Best value before the recent window
        prev_best = min(t.value for t in completed[:-patience])
        # Best value inside the recent window
        recent_best = min(t.value for t in completed[-patience:])

        # Improvement threshold
        thr = max(abs_tol, rel_tol * max(abs(prev_best), 1.0))

        # If the improvement from `prev_best` to `recent_best` is smaller
        # than `thr`, stop (for `minimize`: smaller is better).
        if (prev_best - recent_best) <= thr:
            study.stop()

    return _cb


def make_span_cb(
    window: int = 50,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
):
    """
    If, for the last `window` values, the span (max - min) becomes very small,
    we assume the objective values have essentially flattened out → stop.
    """

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        vals = [t.value for t in study.trials if t.value is not None]
        if len(vals) < window:
            return

        recent = vals[-window:]
        vmin, vmax = min(recent), max(recent)
        span = vmax - vmin
        thr = max(abs_tol, rel_tol * max(abs(vmax), 1.0))

        if span <= thr:
            study.stop()

    return _cb


def make_progress_cb(total_trials: int, every: int = 10):
    """
    A simple callback for printing progress to the console.
    Prints one log line every `every` completed trials.
    """

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        done = len([
            t for t in study.trials
            if t.value is not None or t.state.name != "RUNNING"
        ])

        if done % every != 0 and done != total_trials:
            return

        completed_vals = [
            t.value for t in study.trials
            if t.state.name == "COMPLETE" and t.value is not None
        ]
        if completed_vals:
            # because we are minimizing
            best = min(completed_vals)
            best_str = f"{best:.6f}"
        else:
            best_str = "n/a"

        last = trial.value
        last_str = (
            f"{last:.6f}"
            if isinstance(last, (int, float)) and last == last
            else "n/a"
        )
        print(
            f"[Trial {done}/{total_trials}] "
            f"best={best_str}  last={last_str}  state={trial.state.name}"
        )

    return _cb


# ---------- Generic optimizer wrapper (optimizer core) ----------


def run_optuna_optimizer(
    objective_function: Callable[[Dict[str, float]], float],
    search_space: Dict[str, Tuple[float, float]],
    n_trials: int = 200,
    seed: Optional[int] = None,
    stop_mode: str = "Plateau",  # "Plateau" or "Span"
    min_trials: int = 50,
    patience: int = 20,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
):
    """
    objective_function:
        A callable that takes a dictionary of parameters and returns
        a scalar value. It must be defined such that "smaller is better"
        (we use `direction="minimize"`).

    search_space:
        Dictionary describing the search space:
            {
                "x": (x_min, x_max),
                "y": (y_min, y_max),
                ...
            }

    Returns:
        study, best_params, best_value
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # Sample parameters from the search space
        params: Dict[str, float] = {}
        for name, bounds in search_space.items():
            low, high = bounds
            params[name] = trial.suggest_float(name, float(low), float(high))
        value = float(objective_function(params))
        # Optuna will minimize this value (direction="minimize")
        return value

    # TPE sampler similar to the one used in your code, with some useful settings
    sampler = optuna.samplers.TPESampler(
        seed=int(seed) if seed is not None else None,
        multivariate=True,
        n_startup_trials=20,
        n_ei_candidates=50,
        group=True,  # If you use an older Optuna version and it fails, remove this argument.
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    # Choose the early-stopping callback
    if (stop_mode or "Plateau").lower().startswith("plat"):
        cb_early = make_plateau_cb(
            min_trials=min_trials,
            patience=patience,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    else:
        cb_early = make_span_cb(
            window=min_trials,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    cb_prog = make_progress_cb(
        total_trials=n_trials,
        every=max(n_trials // 20, 1)  # Roughly 20 log lines over the whole run
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[cb_early, cb_prog],
        show_progress_bar=False,
    )

    best_params = study.best_trial.params
    best_value = study.best_value
    return study, best_params, best_value


# ---------- Example usage; similar to ABC.py ----------

if __name__ == "__main__":
    # 1) Define the objective function
    # Here we use a simple test objective (originally from an ABC-style example):
    #   f(x) = |14.5 x - 16|
    # Goal: minimize this value.
    def objective_function(params: Dict[str, float]) -> float:
        x = params["x"]
        Y = params["Y"]

        return float(np.abs(10 * x + 5) - (10 * Y - 20))

    # 2) Define the search space
    search_space = {
        "x": (-50.0, 50.0),  # x_min, x_max
        "Y": (-50.0, 50.0),  # Y_min, Y_max (tune bounds as needed)
    }

    # 3) Run the optimizer
    study, best_params, best_value = run_optuna_optimizer(
        objective_function=objective_function,  # Objective function: takes a dict of parameters and returns a scalar; the optimizer tries to minimize this value.
        search_space=search_space,              # Search space: defines the allowed range for each parameter.
        n_trials=400,                           # Number of trials: maximum number of parameter combinations Optuna will evaluate. More trials = better search but longer runtime.
        seed=42,                                # Random seed for reproducibility. Re-running with the same seed and settings should give very similar results.
        stop_mode="Plateau",                    # Early-stopping mode: "Plateau" stops if there is no improvement for a while; "Span" stops if recent values are very close to each other.
        min_trials=50,                          # Minimum number of trials before we even check early stopping. For "Plateau" this is the warm-up; for "Span" this is the window size.
        patience=20,                            # Used only in "Plateau" mode: if there is no significant improvement in the last 20 trials, stop.
        rel_tol=1e-3,                           # Relative tolerance for improvement: if improvement is less than about 0.1% of the previous value, we treat it as no improvement.
        abs_tol=1e-4,                           # Absolute tolerance: minimum absolute improvement. In practice we use thr = max(abs_tol, rel_tol * |value|).
    )

    # 4) Print final result
    print("\n========== RESULT ==========")
    print("Best params:", best_params)
    print("Best objective value:", best_value)

    # 5) Plot convergence curve (best value so far vs trial index)
    values = [
        t.value
        for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None
    ]

    if values:
        values = np.array(values, dtype=float)
        # Because we are minimizing, "best so far" is the cumulative minimum
        best_so_far = np.minimum.accumulate(values)

        plt.figure(figsize=(6, 4))
        plt.plot(best_so_far, label="Best objective so far")
        plt.xlabel("Completed trials")
        plt.ylabel("Objective value")
        plt.title("Optuna Optimizer Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
