"""Tabular ML models (LightGBM / sklearn fallback) with optional SHAP."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

    HAS_LGBM = True
except Exception:  # noqa: BLE001
    HAS_LGBM = False
    LGBMRanker = None  # type: ignore

from nse_predictor.features.registry import ALL_COMPUTED, available_features

FEATURE_COLS = list(ALL_COMPUTED)
ABSTAIN_BAND = 0.05
THRESH_LO, THRESH_HI = 0.35, 0.65
DEFAULT_LGBM_CLF = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}
DEFAULT_LGBM_RANKER = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def _excess_sample_weight(y_xs: pd.Series, floor: float = 0.25) -> np.ndarray:
    """Upweight large |excess| moves within the labeled set."""
    a = np.abs(y_xs.astype(float).values)
    a = np.nan_to_num(a, nan=0.0)
    if a.max() <= 0:
        return np.ones(len(a))
    w = floor + (1.0 - floor) * (a / (a.max() + 1e-12))
    return w.astype(float)


def labels_from_proba(
    proba: np.ndarray,
    threshold: float = 0.5,
    abstain_band: float = ABSTAIN_BAND,
    mode: str = "threshold",
    trade_dates: pd.Series | None = None,
) -> np.ndarray:
    """Return up/down/flat labels."""
    proba = np.asarray(proba, dtype=float)
    if mode == "ranked" and trade_dates is not None:
        s = pd.Series(proba)
        dt = pd.Series(trade_dates).reset_index(drop=True)
        med = s.groupby(dt).transform("median")
        labels = np.where(s.values >= med.values, "up", "down")
    else:
        labels = np.where(proba >= threshold, "up", "down")
    flat = np.abs(proba - 0.5) < abstain_band
    return np.where(flat, "flat", labels)


def _fit_isotonic_guarded(raw: np.ndarray, y: np.ndarray) -> tuple[IsotonicRegression | None, str]:
    """Fit isotonic; reject if collapsed (std < 0.02 or < 5 unique probs)."""
    if len(y) < 50:
        return None, "none"
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y)
    cal = np.asarray(iso.predict(raw), dtype=float)
    n_unique = int(pd.Series(np.round(cal, 4)).nunique())
    if float(np.std(cal)) < 0.02 or n_unique < 5:
        return None, "rejected_collapse"
    return iso, "isotonic"


def _imputer_model_pipe(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


class _DirectionDecisionMixin:
    """Shared calibrate / threshold / predict frame for binary + ranker heads."""

    model_name: str
    features: list[str]
    threshold_: float
    abstain_band_: float
    decision_mode_: str
    calibrator_: IsotonicRegression | None
    calibration_status_: str
    default_decision_mode: str = "threshold"

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def calibrate(self, df: pd.DataFrame):
        mask = df["y_dir_5d"].notna()
        raw = self.raw_proba(df.loc[mask])
        y = df.loc[mask, "y_dir_5d"].astype(int).values
        self.calibrator_, self.calibration_status_ = _fit_isotonic_guarded(raw, y)
        return self

    def apply_proba(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator_ is None:
            return np.asarray(raw, dtype=float)
        return np.asarray(self.calibrator_.predict(raw), dtype=float)

    def set_threshold(self, threshold: float):
        self.threshold_ = float(np.clip(threshold, THRESH_LO, THRESH_HI))
        return self

    def set_decision_mode(self, mode: str):
        self.decision_mode_ = mode if mode in ("threshold", "ranked") else self.default_decision_mode
        return self

    def copy_decision_from(self, other) -> None:
        self.calibrator_ = other.calibrator_
        self.calibration_status_ = other.calibration_status_
        self.set_threshold(other.threshold_)
        self.set_decision_mode(other.decision_mode_)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.raw_proba(df)
        proba = self.apply_proba(raw)
        labels = labels_from_proba(
            proba,
            threshold=self.threshold_,
            abstain_band=self.abstain_band_,
            mode=self.decision_mode_,
            trade_dates=df["trade_date"],
        )
        out = df[["trade_date", "stock_code"]].copy()
        out["prob_up"] = proba
        out["prob_up_raw"] = raw
        out["pred_label"] = labels
        out["model_name"] = self.model_name
        out["threshold"] = self.threshold_
        out["decision_mode"] = self.decision_mode_
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        return joblib.load(path)


class DirectionModel(_DirectionDecisionMixin):
    default_decision_mode = "threshold"

    def __init__(
        self,
        model_name: str = "lightgbm_direction",
        feature_cols: list[str] | None = None,
        params: dict | None = None,
        abstain_band: float = ABSTAIN_BAND,
    ):
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.params = {**DEFAULT_LGBM_CLF, **(params or {})}
        self.features: list[str] = []
        self.threshold_: float = 0.5
        self.abstain_band_: float = float(abstain_band)
        self.decision_mode_: str = "threshold"
        self.calibrator_: IsotonicRegression | None = None
        self.calibration_status_: str = "none"
        self.scale_pos_weight_: float = 1.0
        self.best_iteration_: int | None = None
        self.pipe = self._make_pipe()

    def _make_pipe(self, scale_pos_weight: float = 1.0) -> Pipeline:
        if HAS_LGBM:
            p = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
            p["scale_pos_weight"] = scale_pos_weight
            clf = LGBMClassifier(**p)
        else:
            clf = GradientBoostingClassifier(random_state=42)
        return _imputer_model_pipe(clf)

    def fit(
        self,
        df: pd.DataFrame,
        balance_classes: bool = True,
        sample_weight_from_xs: bool = True,
        eval_df: pd.DataFrame | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> "DirectionModel":
        self.features = available_features(df, feature_cols=self.feature_cols)
        if not self.features:
            raise ValueError("No model features available after gating")
        mask = df["y_dir_5d"].notna()
        X = df.loc[mask, self.features]
        y = df.loc[mask, "y_dir_5d"].astype(int)

        spw = 1.0
        if balance_classes and len(y):
            pos = float(y.sum())
            neg = float(len(y) - pos)
            spw = float(neg / max(pos, 1.0))
        self.scale_pos_weight_ = spw
        self.pipe = self._make_pipe(scale_pos_weight=spw)

        sw = None
        if sample_weight_from_xs and "y_ret_5d_xs" in df.columns:
            sw = _excess_sample_weight(df.loc[mask, "y_ret_5d_xs"])

        model = self.pipe.named_steps["model"]
        use_es = (
            HAS_LGBM
            and early_stopping_rounds
            and eval_df is not None
            and eval_df["y_dir_5d"].notna().sum() >= 100
        )
        if use_es:
            X_fit = self.pipe.named_steps["imputer"].fit_transform(X)
            ev = eval_df.dropna(subset=["y_dir_5d"])
            X_ev = self.pipe.named_steps["imputer"].transform(ev[self.features])
            fit_kw: dict = {"eval_set": [(X_ev, ev["y_dir_5d"].astype(int))], "callbacks": []}
            try:
                from lightgbm import early_stopping, log_evaluation

                fit_kw["callbacks"] = [
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(period=0),
                ]
            except Exception:  # noqa: BLE001
                pass
            if sw is not None:
                fit_kw["sample_weight"] = sw
            model.fit(X_fit, y, **fit_kw)
            self.best_iteration_ = getattr(model, "best_iteration_", None)
            return self

        if sw is not None:
            X_fit = self.pipe.named_steps["imputer"].fit_transform(X)
            try:
                model.fit(X_fit, y, sample_weight=sw)
            except TypeError:
                model.fit(X_fit, y)
        else:
            self.pipe.fit(X, y)
        return self

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features]
        if HAS_LGBM and hasattr(self.pipe.named_steps["model"], "predict_proba"):
            Xi = self.pipe.named_steps["imputer"].transform(X)
            return self.pipe.named_steps["model"].predict_proba(Xi)[:, 1]
        return self.pipe.predict_proba(X)[:, 1]

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


class DirectionRankModel(_DirectionDecisionMixin):
    """Cross-sectional LGBMRanker; prob_up = within-day percentile of score."""

    default_decision_mode = "ranked"

    def __init__(
        self,
        model_name: str = "lightgbm_direction_ranker",
        feature_cols: list[str] | None = None,
        params: dict | None = None,
        abstain_band: float = ABSTAIN_BAND,
    ):
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.params = {**DEFAULT_LGBM_RANKER, **(params or {})}
        self.features: list[str] = []
        self.threshold_: float = 0.5
        self.abstain_band_: float = float(abstain_band)
        self.decision_mode_: str = "ranked"
        self.calibrator_: IsotonicRegression | None = None
        self.calibration_status_: str = "none"
        self.imputer_ = SimpleImputer(strategy="median")
        self.model_ = None
        self.best_iteration_: int | None = None

    def fit(self, df: pd.DataFrame, **_kwargs) -> "DirectionRankModel":
        if not HAS_LGBM or LGBMRanker is None:
            raise RuntimeError("LGBMRanker unavailable")
        self.features = available_features(df, feature_cols=self.feature_cols)
        d = df.dropna(subset=["y_ret_5d_xs", "trade_date"]).copy()
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        d = d.sort_values(["trade_date", "stock_code"])
        pct = d.groupby("trade_date")["y_ret_5d_xs"].rank(method="average", pct=True)
        d["relevance"] = (pct * 2).round().clip(0, 2).astype(int)
        groups = d.groupby("trade_date", sort=True).size().tolist()
        X = self.imputer_.fit_transform(d[self.features])
        self.model_ = LGBMRanker(objective="lambdarank", **self.params)
        self.model_.fit(X, d["relevance"].values, group=groups)
        return self

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.imputer_.transform(df[self.features])
        return np.asarray(self.model_.predict(X), dtype=float)

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.raw_score(df)
        dt = pd.to_datetime(df["trade_date"]).reset_index(drop=True)
        return pd.Series(scores).groupby(dt).rank(pct=True).values.astype(float)

    def shap_summary(self, df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame | None:
        return None


class ReturnIntervalModel:
    """Predict 5d excess return + residual quantile interval."""

    def __init__(
        self,
        model_name: str = "lightgbm_return_5d_xs",
        feature_cols: list[str] | None = None,
        target_col: str = "y_ret_5d_xs",
        params: dict | None = None,
    ):
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
            **(params or {}),
        }
        self.features: list[str] = []
        self.residual_std_: float = 0.03
        self.q_low_: float = -0.03
        self.q_high_: float = 0.03
        self.quantile_low_p_: float = 0.10
        self.quantile_high_p_: float = 0.90
        self.best_iteration_: int | None = None
        reg = LGBMRegressor(**self.params) if HAS_LGBM else GradientBoostingRegressor(random_state=42)
        self.pipe = _imputer_model_pipe(reg)

    def fit(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> "ReturnIntervalModel":
        self.features = available_features(df, feature_cols=self.feature_cols)
        if not self.features:
            raise ValueError("No model features available after gating")
        target = self.target_col if self.target_col in df.columns else "y_ret_5d"
        self.target_col = target
        mask = df[target].notna()
        X = df.loc[mask, self.features]
        y = df.loc[mask, target].astype(float)
        self.pipe.fit(X, y)

        if (
            HAS_LGBM
            and early_stopping_rounds
            and eval_df is not None
            and eval_df[target].notna().sum() >= 100
        ):
            try:
                from lightgbm import early_stopping, log_evaluation

                model = self.pipe.named_steps["model"]
                X_fit = self.pipe.named_steps["imputer"].transform(X)
                ev = eval_df.dropna(subset=[target])
                X_ev = self.pipe.named_steps["imputer"].transform(ev[self.features])
                model.fit(
                    X_fit,
                    y,
                    eval_set=[(X_ev, ev[target].astype(float))],
                    callbacks=[
                        early_stopping(early_stopping_rounds, verbose=False),
                        log_evaluation(period=0),
                    ],
                )
                self.best_iteration_ = getattr(model, "best_iteration_", None)
            except Exception:  # noqa: BLE001
                pass

        resid = y.values - self.pipe.predict(X)
        self.residual_std_ = float(np.std(resid)) or 0.03
        self.q_low_, self.q_high_ = -self.residual_std_, self.residual_std_
        return self

    def set_residual_quantiles(
        self,
        df: pd.DataFrame,
        q_low: float = 0.10,
        q_high: float = 0.90,
    ) -> "ReturnIntervalModel":
        mask = df[self.target_col].notna()
        y = df.loc[mask, self.target_col].astype(float).values
        pred = self.pipe.predict(df.loc[mask, self.features])
        resid = y - pred
        self.quantile_low_p_, self.quantile_high_p_ = float(q_low), float(q_high)
        self.q_low_ = float(np.quantile(resid, q_low))
        self.q_high_ = float(np.quantile(resid, q_high))
        self.residual_std_ = float(np.std(resid)) or self.residual_std_
        return self

    def copy_interval_from(self, other: "ReturnIntervalModel") -> "ReturnIntervalModel":
        self.q_low_ = other.q_low_
        self.q_high_ = other.q_high_
        self.quantile_low_p_ = other.quantile_low_p_
        self.quantile_high_p_ = other.quantile_high_p_
        self.residual_std_ = other.residual_std_
        return self

    def predict(self, df: pd.DataFrame, z: float | None = None) -> pd.DataFrame:
        pred = self.pipe.predict(df[self.features])
        out = df[["trade_date", "stock_code"]].copy()
        out["pred_value"] = pred
        if z is None:
            out["lower"] = pred + self.q_low_
            out["upper"] = pred + self.q_high_
        else:
            out["lower"] = pred - z * self.residual_std_
            out["upper"] = pred + z * self.residual_std_
        out["model_name"] = self.model_name
        out["target_col"] = self.target_col
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ReturnIntervalModel":
        return joblib.load(path)


class VolatilityModel:
    def __init__(
        self,
        model_name: str = "lightgbm_vol_20d",
        feature_cols: list[str] | None = None,
        params: dict | None = None,
    ):
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
            **(params or {}),
        }
        self.features: list[str] = []
        self.best_iteration_: int | None = None
        reg = LGBMRegressor(**self.params) if HAS_LGBM else GradientBoostingRegressor(random_state=42)
        self.pipe = _imputer_model_pipe(reg)

    def fit(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> "VolatilityModel":
        self.features = available_features(df, feature_cols=self.feature_cols)
        if not self.features:
            raise ValueError("No model features available after gating")
        mask = df["y_vol_20d"].notna()
        self.pipe.fit(df.loc[mask, self.features], df.loc[mask, "y_vol_20d"].astype(float))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["trade_date", "stock_code"]].copy()
        out["pred_value"] = np.clip(self.pipe.predict(df[self.features]), 0, None)
        out["model_name"] = self.model_name
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "VolatilityModel":
        return joblib.load(path)
