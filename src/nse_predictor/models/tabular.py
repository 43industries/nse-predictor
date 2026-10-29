"""Tabular ML models (LightGBM / sklearn fallback) with optional SHAP."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True
except Exception:  # noqa: BLE001
    HAS_LGBM = False

FEATURE_COLS = [
    "ret_1d",
    "ma5_ratio",
    "ma20_ratio",
    "ma50_ratio",
    "vol_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "dist_52w_high",
    "dist_52w_low",
    "volume_chg",
    "days_since_trade",
    "nasi_ret",
    "sector_ret",
    "cbk_rate",
    "cbk_chg",
    "tbill_91d",
    "usd_kes",
    "usd_kes_chg",
    "inflation_yoy",
    "inflation_trend",
    "lending_rate",
    "dow",
    "month",
    "results_season",
]


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


class DirectionModel:
    def __init__(self, model_name: str = "lightgbm_direction"):
        self.model_name = model_name
        self.features: list[str] = []
        if HAS_LGBM:
            clf = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        else:
            clf = GradientBoostingClassifier(random_state=42)
        self.pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", clf),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "DirectionModel":
        self.features = available_features(df)
        mask = df["y_dir_5d"].notna()
        X = df.loc[mask, self.features]
        y = df.loc[mask, "y_dir_5d"].astype(int)
        self.pipe.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        proba = self.pipe.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out = df[["trade_date", "stock_code"]].copy()
        out["prob_up"] = proba
        out["pred_label"] = np.where(pred == 1, "up", "down")
        out["model_name"] = self.model_name
        return out

    def shap_summary(self, df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame | None:
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        sample = df[self.features].tail(max_rows)
        X = self.pipe.named_steps["imputer"].transform(sample)
        model = self.pipe.named_steps["model"]
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            if isinstance(values, list):
                values = values[1]
            mean_abs = np.abs(values).mean(axis=0)
            return (
                pd.DataFrame({"feature": self.features, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
        except Exception:  # noqa: BLE001
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "DirectionModel":
        return joblib.load(path)


class ReturnIntervalModel:
    """Predict 5d return + residual-based interval."""

    def __init__(self, model_name: str = "lightgbm_return_5d"):
        self.model_name = model_name
        self.features: list[str] = []
        self.residual_std_: float = 0.03
        if HAS_LGBM:
            reg = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
        else:
            reg = GradientBoostingRegressor(random_state=42)
        self.pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", reg),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "ReturnIntervalModel":
        self.features = available_features(df)
        mask = df["y_ret_5d"].notna()
        X = df.loc[mask, self.features]
        y = df.loc[mask, "y_ret_5d"].astype(float)
        self.pipe.fit(X, y)
        pred = self.pipe.predict(X)
        self.residual_std_ = float(np.std(y - pred)) or 0.03
        return self

    def predict(self, df: pd.DataFrame, z: float = 1.0) -> pd.DataFrame:
        pred = self.pipe.predict(df[self.features])
        out = df[["trade_date", "stock_code"]].copy()
        out["pred_value"] = pred
        out["lower"] = pred - z * self.residual_std_
        out["upper"] = pred + z * self.residual_std_
        out["model_name"] = self.model_name
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ReturnIntervalModel":
        return joblib.load(path)


class VolatilityModel:
    def __init__(self, model_name: str = "lightgbm_vol_20d"):
        self.model_name = model_name
        self.features: list[str] = []
        if HAS_LGBM:
            reg = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        else:
            reg = GradientBoostingRegressor(random_state=42)
        self.pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", reg),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "VolatilityModel":
        self.features = available_features(df)
        mask = df["y_vol_20d"].notna()
        X = df.loc[mask, self.features]
        y = df.loc[mask, "y_vol_20d"].astype(float)
        self.pipe.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pred = self.pipe.predict(df[self.features])
        out = df[["trade_date", "stock_code"]].copy()
        out["pred_value"] = np.clip(pred, 0, None)
        out["model_name"] = self.model_name
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "VolatilityModel":
        return joblib.load(path)
