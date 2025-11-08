import numpy as np
import optuna
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional

# ---------- Early stopping callbacks (استخراج‌شده و ساده‌شده) ----------

def make_plateau_cb(
    min_trials: int = 50,
    patience: int = 20,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
):
    """
    اگر بعد از حداقل min_trials، در پنجره‌ی آخر به طول patience
    بهبود قابل توجهی (بزرگ‌تر از آستانه) رخ ندهد، study را متوقف می‌کند.
    برای حالت minimize نوشته شده است.
    """
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed = [
            t for t in study.trials
            if (t.value is not None and t.state.name == "COMPLETE")
        ]
        if len(completed) < (min_trials + patience):
            return

        # بهترین مقدار قبل از پنجره‌ی اخیر
        prev_best = min(t.value for t in completed[:-patience])
        # بهترین مقدار داخل پنجره‌ی اخیر
        recent_best = min(t.value for t in completed[-patience:])

        # آستانه بهبود
        thr = max(abs_tol, rel_tol * max(abs(prev_best), 1.0))

        # اگر بهبودِ prev_best → recent_best کمتر از thr باشد متوقف کن
        # (برای minimize: هر چه کوچک‌تر بهتر)
        if (prev_best - recent_best) <= thr:
            study.stop()

    return _cb


def make_span_cb(
    window: int = 50,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
):
    """
    اگر در آخرین window مقدار، بازه‌ی (max - min) خیلی کوچک باشد،
    یعنی مقادیر تقریباً تخت شده‌اند → متوقف کن.
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
    یک کال‌بک ساده برای نمایش پیشرفت روی کنسول.
    هر every تریال یک خط چاپ می‌کند.
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
            best = min(completed_vals)  # چون داریم minimize می‌کنیم
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


# ---------- Generic optimizer wrapper (هسته اپتیمایزر) ----------

def run_optuna_optimizer(
    objective_function: Callable[[Dict[str, float]], float],
    search_space: Dict[str, Tuple[float, float]],
    n_trials: int = 200,
    seed: Optional[int] = None,
    stop_mode: str = "Plateau",  # "Plateau" یا "Span"
    min_trials: int = 50,
    patience: int = 20,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
):
    """
    objective_function:
        تابع هدفی که یک دیکشنری از پارامترها را می‌گیرد و مقدار اسکالر برمی‌گرداند.
        این تابع باید طوری تعریف شود که «هر چه مقدارش کمتر باشد، بهتر است» (minimize).

    search_space:
        دیکشنری فضای جستجو:
            {
                "x": (x_min, x_max),
                "y": (y_min, y_max),
                ...
            }

    خروجی:
        study, best_params, best_value
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # نمونه‌برداری از فضای جستجو
        params: Dict[str, float] = {}
        for name, bounds in search_space.items():
            low, high = bounds
            params[name] = trial.suggest_float(name, float(low), float(high))
        value = float(objective_function(params))
        return value  # Optuna آن را مینیمم می‌کند (direction="minimize")

    # همان TPE Sampler که در کدت استفاده کرده‌ای، با چند تنظیم مفید
    sampler = optuna.samplers.TPESampler(
        seed=int(seed) if seed is not None else None,
        multivariate=True,
        n_startup_trials=20,
        n_ei_candidates=50,
        group=True,  # اگر نسخه Optuna قدیمی بود و خطا داد، این آرگومان را حذف کن.
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    # انتخاب کال‌بک توقف زودهنگام
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
        every=max(n_trials // 20, 1)  # حدوداً ۲۰ لاگ در طول کار
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


# ---------- مثال استفاده؛ شبیه ABC.py ----------

if __name__ == "__main__":
    # ۱) تعریف تابع هدف
    # اینجا همان تابع ABC را استفاده کرده‌ام:
    #   f(x) = |14.5 x - 16|
    # هدف: کمینه کردن این مقدار.
    def objective_function(params: Dict[str, float]) -> float:
        x = params["x"]
        Y = params["Y"]

        return float(np.abs(10 * x + 5) - (10 * Y - 20))

    # ۲) تعریف فضای جستجو (مثلاً فقط یک متغیر x)
    # ۲) تعریف فضای جستجو
    search_space = {
        "x": (-50.0, 50.0),  # x_min, x_max
        "Y": (-50.0, 50.0),  # Y_min, Y_max  ← این را اضافه کن (بازه را هرطور لازم است تنظیم کن)
    }

    # ۳) اجرای اپتیمایزر
    study, best_params, best_value = run_optuna_optimizer(
        objective_function=objective_function,  # تابع هدف: ورودی‌اش دیکشنری پارامترهاست و یک عدد برمی‌گرداند؛ اپتیمایزر سعی می‌کند این عدد را «کمینه یا بیشینه» کند.
        search_space=search_space,              # فضای جست‌وجو: مشخص می‌کند هر پارامتر در چه بازه‌ای می‌تواند باشد.
        n_trials=400,                           # تعداد تریال‌ها (آزمایش‌ها): حداکثر چند بار Optuna ترکیب‌های مختلف پارامترها را امتحان کند. هرچه بیشتر، جست‌وجو دقیق‌تر ولی زمان‌برتر.
        seed=42,                                # بذر تصادفی (random seed): برای تکرارپذیری. اگر دفعه‌های بعد هم با همین seed و تنظیمات اجرا کنی،تقریباً همان جواب‌ها را می‌گیری.
        stop_mode="Plateau",                    # نوع توقف زودهنگام:"Plateau": اگر مدتی بهبودی در نتیجه دیده نشود، متوقف شود. "Span": اگر مقادیر اخیر خیلی به هم نزدیک شوند (نوسان کم)، متوقف شود.
        min_trials=50,                          # حداقل تعداد تریالی که قبل از بررسی توقف باید انجام شود.در حالت Plateau: تا قبل از 50 تریال، اصلاً به توقف فکر نمی‌کند.در حالت Span: اندازه‌ی پنجره‌ای که آخرین مقادیر را بررسی می‌کند.
        patience=20,                            # مخصوص حالت Plateau:اگر در 20 تریال آخر، بهبود معنی‌دار نسبت به قبل رخ ندهد،اپتیمایزر را متوقف می‌کند.
        rel_tol=1e-3,                           # آستانه‌ی بهبود نسبی (relative tolerance):یعنی اگر بهبود کمتر از حدود 0.1٪ مقدار قبلی باشد،حساب می‌شود که «دیگر بهبود خاصی نشده».
        abs_tol=1e-4,                           # آستانه‌ی بهبود مطلق (absolute tolerance): حداقل بهبود بر حسب مقدار مطلق. اگر فرق قدیم و جدید از این هم کمتر باشد، یعنی عملاً تغییری نکرده.در عمل، thr = max(abs_tol, rel_tol * مقدار) استفاده می‌شود.
    )

    # ۴) چاپ نتیجه نهایی
    print("\n========== RESULT ==========")
    print("Best params:", best_params)
    print("Best objective value:", best_value)

    # ۵) رسم نمودار همگرایی (بهترین مقدار تاکنون نسبت به شماره تریال)
    values = [
        t.value
        for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None
    ]

    if values:
        values = np.array(values, dtype=float)
        # چون minimize است، best-so-far یعنی مینیمم تجمعی
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
