"""T-learner: two outcome models, one per arm.

Chosen over a single S-learner because the treatment effect here is expected to
be heterogeneous and the sample is small; two simpler models are easier to
diagnose than one model with interaction terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _logistic() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]
    )


def _gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.06,
        max_iter=250,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


@dataclass
class TLearner:
    """Two outcome models, one per arm.

    `kind` lets the same pipeline compare a smooth baseline with a tree model;
    the tree is the default because the response is driven by level × tier
    interactions that a linear model cannot express.
    """

    kind: str = "gbm"
    C: float = 0.5

    def __post_init__(self) -> None:
        self.treated_model = None
        self.control_model = None
        self.columns: list[str] = []

    def _new(self):
        return _gbm() if self.kind == "gbm" else _logistic()

    def fit(self, X: pd.DataFrame, treated: pd.Series, y: pd.Series) -> TLearner:
        self.columns = list(X.columns)
        mask_t = treated.to_numpy() == 1
        self.treated_model = self._new()
        self.control_model = self._new()
        self.treated_model.fit(X[mask_t], y[mask_t])
        self.control_model.fit(X[~mask_t], y[~mask_t])
        return self

    def predict_arms(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.treated_model is None or self.control_model is None:
            raise RuntimeError("model is not fitted")
        X = X[self.columns]
        return (
            self.treated_model.predict_proba(X)[:, 1],
            self.control_model.predict_proba(X)[:, 1],
        )

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p1, p0 = self.predict_arms(X)
        return p1 - p0


def naive_rate(treated: pd.Series, y: pd.Series) -> dict[str, float]:
    """Headline read-out, deliberately shown next to uplift to expose the gap."""
    t = treated.to_numpy() == 1
    yv = y.to_numpy()
    p1 = float(yv[t].mean()) if t.any() else 0.0
    p0 = float(yv[~t].mean()) if (~t).any() else 0.0
    return {"treated_rate": round(p1, 4), "control_rate": round(p0, 4), "naive_lift": round(p1 - p0, 4)}


@dataclass
class TransformedOutcome:
    """One model on the transformed label z = y * (t/e - (1-t)/(1-e)).

    With e = P(treated) = 0.5, E[z | X] = uplift(X). This uses every row in a
    single model, which matters when each arm is too small to fit separately.
    """

    kind: str = "ridge"
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.model = None
        self.columns: list[str] = []
        self.e = 0.5

    def _new(self):
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge

            return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=self.alpha))])
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=80, random_state=42
        )

    def fit(self, X: pd.DataFrame, treated: pd.Series, y: pd.Series) -> TransformedOutcome:
        self.columns = list(X.columns)
        t = treated.to_numpy().astype(float)
        self.e = float(np.clip(t.mean(), 0.05, 0.95))
        z = y.to_numpy().astype(float) * (t / self.e - (1 - t) / (1 - self.e))
        self.model = self._new()
        self.model.fit(X, z)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(self.model.predict(X[self.columns]), dtype=float)
