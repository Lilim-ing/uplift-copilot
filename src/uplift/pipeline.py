"""Train → evaluate → allocate, in one call."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .features import build_dataset, split_xy
from .metrics import auuc, decile_table, qini_coefficient, uplift_at_k
from .models import TLearner, TransformedOutcome, naive_rate
from .policy import compare_policies
from .warehouse import Warehouse


@dataclass
class RunResult:
    campaigns: list[str]
    n_rows: int
    n_train: int
    n_test: int
    treatment_share: float
    naive: dict = field(default_factory=dict)
    qini: float = 0.0
    auuc: float = 0.0
    uplift_at_10pct: float = 0.0
    deciles: list[dict] = field(default_factory=list)
    allocation: list[dict] = field(default_factory=list)
    budget: float = 0.0
    cost_per_touch: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run(
    data_dir: str | Path,
    campaign_ids: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    budget: float = 50_000,
    cost_per_touch: float = 8.0,
    model_kind: str = "to_ridge",
) -> RunResult:
    with Warehouse(data_dir) as w:
        df = build_dataset(w, campaign_ids)

    X, t, y = split_xy(df)
    # Stratify on the joint label so both arms and both outcomes appear in test.
    strat = (t.astype(str) + "_" + y.astype(str)).values
    X_tr, X_te, t_tr, t_te, y_tr, y_te = train_test_split(
        X, t, y, test_size=test_size, random_state=seed, stratify=strat
    )

    if model_kind == "t_learner":
        model = TLearner(kind="logistic").fit(X_tr, t_tr, y_tr)
        uplift_te = model.predict_uplift(X_te)
        p1_te, _ = model.predict_arms(X_te)
    elif model_kind == "t_learner_gbm":
        model = TLearner(kind="gbm").fit(X_tr, t_tr, y_tr)
        uplift_te = model.predict_uplift(X_te)
        p1_te, _ = model.predict_arms(X_te)
    elif model_kind in {"to_ridge", "to_gbm"}:
        model = TransformedOutcome(kind="ridge" if model_kind == "to_ridge" else "gbm").fit(
            X_tr, t_tr, y_tr
        )
        uplift_te = model.predict_uplift(X_te)
        # A transformed-outcome model predicts uplift directly and has no
        # conversion probability. Do not fake one: omit the comparison arm
        # rather than rank by clipped uplift, which is not a probability.
        p1_te = None
    else:
        raise ValueError(f"unknown model_kind {model_kind!r}")

    # A deliberately dumb benchmark: mean historical uplift per (level, tier).
    # It has no parameters per member, so it cannot overfit a small sample.
    train_meta = df.loc[X_tr.index, ["member_level", "city_tier"]].copy()
    train_meta["treated"] = t_tr.values
    train_meta["converted"] = y_tr.values
    seg = (
        train_meta.groupby(["member_level", "city_tier"])
        .apply(
            lambda g: g.loc[g.treated == 1, "converted"].mean()
            - g.loc[g.treated == 0, "converted"].mean(),
            include_groups=False,
        )
        .rename("segment_uplift")
        .reset_index()
    )
    test_meta = df.loc[X_te.index, ["member_level", "city_tier"]].merge(
        seg, on=["member_level", "city_tier"], how="left"
    )
    segment_score = test_meta["segment_uplift"].fillna(0.0).to_numpy()

    cost = np.full(len(X_te), cost_per_touch)
    allocation = compare_policies(
        uplift_te,
        cost,
        budget,
        baseline_score=p1_te,
        segment_score=segment_score,
        y=y_te.to_numpy(),
        t=t_te.to_numpy(),
    )

    note = ""
    if t_te.nunique() < 2 or y_te.nunique() < 2:
        note = "测试集某一臂没有转化样本，Qini 不可信"
    elif np.allclose(uplift_te, 0):
        note = "模型输出常数，说明特征没有区分度"

    return RunResult(
        campaigns=sorted(df["campaign_id"].unique().tolist()),
        n_rows=len(df),
        n_train=len(X_tr),
        n_test=len(X_te),
        treatment_share=round(float(t.mean()), 4),
        naive=naive_rate(t_te.reset_index(drop=True), y_te.reset_index(drop=True)),
        qini=qini_coefficient(y_te.to_numpy(), t_te.to_numpy(), uplift_te),
        auuc=auuc(y_te.to_numpy(), t_te.to_numpy(), uplift_te),
        uplift_at_10pct=uplift_at_k(y_te.to_numpy(), t_te.to_numpy(), uplift_te, 0.1),
        deciles=decile_table(y_te.to_numpy(), t_te.to_numpy(), uplift_te, 5),
        allocation=allocation,
        budget=budget,
        cost_per_touch=cost_per_touch,
        note=note,
    )
