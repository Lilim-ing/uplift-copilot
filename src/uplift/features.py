"""Member-level feature assembly.

Point-in-time discipline: every behavioural feature is computed from the 90 days
*before* the campaign starts. Using data from after the campaign would leak the
outcome into the features and inflate the uplift estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .warehouse import Warehouse

CATEGORICAL = ["member_level", "city_tier", "age_band", "register_channel", "channel"]
NUMERIC = ["underbust", "orders_90d", "gmv_90d", "aov_90d", "recency_days", "coupon_value"]

SQL = """
WITH agg AS (
  SELECT ca.campaign_id, ca.member_id,
         COUNT(o.order_id) AS orders_90d,
         COALESCE(SUM(o.paid_amount), 0) AS gmv_90d,
         MAX(CAST(o.order_time AS DATE)) AS last_order
  FROM campaign_assignments ca
  JOIN campaigns c USING(campaign_id)
  LEFT JOIN orders o
    ON o.member_id = ca.member_id
   AND o.status <> 'refunded'
   AND CAST(o.order_time AS DATE)
       BETWEEN CAST(c.start_date AS DATE) - INTERVAL 89 DAY AND CAST(c.start_date AS DATE)
  GROUP BY 1, 2
)
SELECT ca.campaign_id, ca.member_id, ca.treated, ca.converted, ca.revenue,
       c.start_date, c.coupon_value, c.channel,
       m.member_level, m.city_tier, m.age_band, m.register_channel, m.underbust, m.cup,
       a.orders_90d, a.gmv_90d, a.last_order
FROM campaign_assignments ca
JOIN campaigns c USING(campaign_id)
JOIN members m USING(member_id)
JOIN agg a USING(campaign_id, member_id)
ORDER BY ca.campaign_id, ca.member_id
"""


def build_dataset(w: Warehouse, campaign_ids: list[str] | None = None) -> pd.DataFrame:
    df = w.query(SQL)
    if campaign_ids:
        df = df[df["campaign_id"].isin(campaign_ids)].copy()

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["last_order"] = pd.to_datetime(df["last_order"])
    # Never purchased in the window -> long recency, not a missing value.
    df["recency_days"] = (df["start_date"] - df["last_order"]).dt.days.fillna(999).astype(int)
    df["aov_90d"] = np.where(
        df["orders_90d"] > 0, df["gmv_90d"] / df["orders_90d"].clip(lower=1), 0.0
    )
    return df.reset_index(drop=True)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals; return (matrix, feature_names)."""
    cats = pd.get_dummies(df[CATEGORICAL].astype(str), drop_first=True)
    num = df[NUMERIC].astype(float)
    X = pd.concat([num, cats], axis=1)
    return X, list(X.columns)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, _ = encode(df)
    return X, df["treated"].astype(int), df["converted"].astype(int)
