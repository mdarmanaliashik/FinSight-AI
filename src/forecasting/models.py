"""
FinSight AI - Forecasting Model Suite: Baselines & Quantile LightGBM
=====================================================================
Q1 Publication & Competition Grade Forecasting Implementations

Models:
- A0: Last-Value Naive
- A1: Seasonal Naive (7-day periodic lag)
- A2: Exponential Moving Average / Moving Average (28-day window)
- A3: LightGBM Multi-Quantile Regressor (P10, P50, P90) with Recursive Multi-Step Rollout
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.forecasting.features import extract_features_from_history, FEATURE_COLS


class LastValueNaive:
    """
    Model A0: Projects the most recently observed value forward.
    """
    def __init__(self):
        self.last_value_ = None

    def fit(self, y_train: np.ndarray):
        self.last_value_ = float(y_train[-1])
        return self

    def predict(self, horizon: int, last_observed: Optional[float] = None) -> np.ndarray:
        val = float(last_observed) if last_observed is not None else self.last_value_
        return np.full(horizon, val, dtype=float)


class SeasonalNaive:
    """
    Model A1: Repeats the observed values from the same day of the week
    in the preceding cycle (7-day seasonal lag).
    """
    def __init__(self, seasonal_period: int = 7):
        self.period = seasonal_period
        self.history_ = None

    def fit(self, y_train: np.ndarray):
        if len(y_train) < self.period:
            raise ValueError(f"History length must be at least {self.period}")
        self.history_ = np.asarray(y_train, dtype=float)
        return self

    def predict(self, horizon: int, recent_history: Optional[np.ndarray] = None) -> np.ndarray:
        hist = np.asarray(recent_history, dtype=float) if recent_history is not None else self.history_
        last_cycle = hist[-self.period:]
        reps = int(np.ceil(horizon / self.period))
        tiled = np.tile(last_cycle, reps)
        return tiled[:horizon]


class MovingAverageBaseline:
    """
    Model A2: Exponential Moving Average (EMA) or Rolling MA over past W days.
    """
    def __init__(self, window: int = 28, alpha: Optional[float] = None):
        self.window = window
        self.alpha = alpha if alpha is not None else (2.0 / (window + 1))
        self.val_ = None

    def fit(self, y_train: np.ndarray):
        w = min(self.window, len(y_train))
        # Compute EMA over training series
        series = pd.Series(y_train)
        self.val_ = float(series.ewm(span=self.window, adjust=False).mean().iloc[-1])
        return self

    def predict(self, horizon: int, recent_history: Optional[np.ndarray] = None) -> np.ndarray:
        if recent_history is not None:
            series = pd.Series(recent_history)
            val = float(series.ewm(span=self.window, adjust=False).mean().iloc[-1])
        else:
            val = self.val_
        return np.full(horizon, val, dtype=float)


class LightGBMQuantileForecaster:
    """
    Model A3: LightGBM Multi-Quantile Regressor.
    Fits gradient-boosted trees for alpha in {0.10, 0.50, 0.90}
    under pinball loss with strict seed control and zero-leakage recursive trajectory rollout.
    """
    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        random_state: int = 42
    ):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.models_: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: List[str] = FEATURE_COLS

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        self.feature_names_ = list(X_train.columns)

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbosity=-1
            )
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
            else:
                model.fit(X_train, y_train)

            self.models_[q] = model

        return self

    def predict(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Direct tabular prediction across test rows.
        """
        preds = {}
        for q, model in self.models_.items():
            key = f"p{int(q * 100)}"
            preds[key] = np.maximum(0.0, model.predict(X_test))

        # Enforce quantile monotonicity: P10 <= P50 <= P90
        if "p10" in preds and "p50" in preds and "p90" in preds:
            preds["p10"] = np.minimum(preds["p10"], preds["p50"])
            preds["p90"] = np.maximum(preds["p90"], preds["p50"])

        return preds

    def predict_trajectory(
        self,
        history_sales: Union[pd.Series, np.ndarray, List[float]],
        start_date: Union[pd.Timestamp, str],
        horizon: int = 21
    ) -> Dict[str, np.ndarray]:
        """
        100% Zero-Leakage Recursive Autoregressive Trajectory Rollout.
        At time origin t, given only history_sales <= t, projects h = 1..horizon steps ahead.
        At each step h, newly predicted values are fed back into features without touching
        any actual future ground-truth values.
        """
        curr_hist = list(np.asarray(history_sales, dtype=float))
        start_dt = pd.to_datetime(start_date)

        p10_list, p50_list, p90_list = [], [], []

        for step in range(horizon):
            target_dt = start_dt + pd.Timedelta(days=step)
            # Extract features purely from current synthetic/historical state
            x_step = extract_features_from_history(curr_hist, target_dt)

            p10_val = float(np.maximum(0.0, self.models_[0.10].predict(x_step)[0])) if 0.10 in self.models_ else 0.0
            p50_val = float(np.maximum(0.0, self.models_[0.50].predict(x_step)[0])) if 0.50 in self.models_ else 0.0
            p90_val = float(np.maximum(0.0, self.models_[0.90].predict(x_step)[0])) if 0.90 in self.models_ else 0.0

            # Monotonicity adjustment
            p10_val = min(p10_val, p50_val)
            p90_val = max(p90_val, p50_val)

            p10_list.append(p10_val)
            p50_list.append(p50_val)
            p90_list.append(p90_val)

            # Autoregressive feedback: append expected P50 prediction to history for next step
            curr_hist.append(p50_val)

        return {
            "p10": np.array(p10_list, dtype=float),
            "p50": np.array(p50_list, dtype=float),
            "p90": np.array(p90_list, dtype=float)
        }

    def get_feature_importance(self) -> pd.DataFrame:
        if 0.50 not in self.models_:
            p50_model = list(self.models_.values())[0]
        else:
            p50_model = self.models_[0.50]

        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": p50_model.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp
