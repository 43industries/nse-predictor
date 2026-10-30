#!/usr/bin/env python3
"""Build features, train with valid selection, evaluate, save artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.baselines.rules import always_up, moving_average_direction
from nse_predictor.config import settings
from nse_predictor.eval.backtest import simulate_book_long_flat
from nse_predictor.eval.diagnostics import (
    binary_from_labels,
    calibration_pack,
    per_stock_direction,
    write_train_val_report,
)
from nse_predictor.eval.error_history import build_error_history
from nse_predictor.eval.gates import compute_validation_gates
from nse_predictor.eval.metrics import majority_baseline_pred, rank_ic, regression_metrics
from nse_predictor.eval.protocol import LABEL_HORIZON_DAYS, time_split_embargoed
from nse_predictor.eval.validate import (
    direction_diagnostics,
    eval_return_model,
    is_stress_fragile,
    map_stock_freq,
    stress_oos_retrain,
    write_validation_report,
)
from nse_predictor.eval.walk_forward import evaluate_direction
from nse_predictor.features.ablation import select_feature_preset, write_ablation_report
from nse_predictor.features.build import DEAD_ZONE_XS
from nse_predictor.features.cache import load_macro_source, load_or_build_features
from nse_predictor.ingest.fundamentals import load_fundamentals
from nse_predictor.models.classical import fit_arima_forecast, fit_garch_vol
from nse_predictor.models.tabular import VolatilityModel
from nse_predictor.serving.score import score_latest_forecasts
from nse_predictor.training.fit import (
    direction_metrics_bundle,
    fit_direction_selected,
    fit_return_selected,
    maybe_add_stock_freq_feature,
    run_walk_forward_direction,
    select_direction_head,
    split_meta,
    valid_rank_ic,
    walk_forward_summary,
)
from nse_predictor.training.meta import write_model_meta
from nse_predictor.universe import investable_equities, pick_universe, save_universe


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = settings.data_processed_dir
    return (
        pd.read_parquet(p / "prices.parquet"),
        pd.read_parquet(p / "macro.parquet"),
        pd.read_parquet(p / "sectors.parquet"),
    )


def _baselines(eval_df: pd.DataFrame) -> dict:
    y = eval_df["y_dir_5d"].astype(int).values
    au = always_up(eval_df)
    ma = moving_average_direction(eval_df)
    out = {
        "always_up": evaluate_direction(y, np.ones_like(y), au["prob_up"].values),
        "majority": evaluate_direction(y, majority_baseline_pred(y)),
        "ma_5_20": evaluate_direction(
            y, (ma["pred_label"] == "up").astype(int).values, ma["prob_up"].values
        ),
        "tomorrow_eq_today_ret5d": regression_metrics(
            eval_df["y_ret_5d"].values, np.zeros(len(eval_df))
        ),
    }
    if "y_ret_5d_xs" in eval_df.columns:
        out["zero_excess_ret5d"] = regression_metrics(
            eval_df["y_ret_5d_xs"].values, np.zeros(len(eval_df))
        )
    return out


def _apply_stock_freq(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    train_ready: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, dict]:
    probe_model, _ = fit_direction_selected(train, valid, feature_cols=feature_cols, tune=False)
    _, probe_pred = direction_metrics_bundle(valid, probe_model)
    probe_ic = valid_rank_ic(valid, probe_pred)
    per_valid = per_stock_direction(valid, probe_pred)
    flagged_frac = sum(1 for r in per_valid if r.get("flag")) / max(len(per_valid), 1)

    feature_cols, train, valid, used, info = maybe_add_stock_freq_feature(
        train, valid, feature_cols, flagged_frac, baseline_rank_ic=probe_ic
    )
    freq_map = info.get("freq_map") or {}
    if used and freq_map:
        test, train_ready = map_stock_freq([test, train_ready], freq_map)
        print(
            f"stock_freq kept (flagged_frac={flagged_frac:.0%}, "
            f"ic {info.get('baseline_rank_ic')} → {info.get('trial_rank_ic')})"
        )
    elif flagged_frac >= 0.4:
        print(f"stock_freq trial skipped/rejected (flagged_frac={flagged_frac:.0%})")
    return feature_cols, train, valid, test, train_ready, used, info


def main() -> int:
    prices, macro, sectors = load_processed()
    universe = pick_universe(prices)
    save_universe(universe)
    investable = investable_equities(universe)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    macro_source = load_macro_source(settings.data_processed_dir)
    fund = load_fundamentals()
    feats, cache_info = load_or_build_features(
        prices, macro, sectors, universe, fundamentals=fund, macro_source=macro_source
    )
    fund_included = bool(
        feats.attrs.get("fundamentals_included") or cache_info.get("fundamentals_included")
    )
    fund_status = feats.attrs.get("fundamentals_status") or cache_info.get("fundamentals_status")
    if str(macro_source).startswith("synthetic"):
        print(f"WARNING: macro_source={macro_source}", file=sys.stderr)

    train_ready = feats.dropna(subset=["y_dir_5d", "ret_1d", "ma5_ratio"])
    train_ready = train_ready[train_ready["stock_code"].isin(investable)]
    train, valid, test, protocol = time_split_embargoed(
        train_ready,
        train_end=settings.train_end,
        valid_end=settings.valid_end,
        horizon_days=LABEL_HORIZON_DAYS,
    )
    print(
        f"protocol embargo={protocol['embargo']} "
        f"train_fit={protocol['n_train_fit']}/{protocol['n_train_raw']} "
        f"valid_fit={protocol['n_valid_fit']}/{protocol['n_valid_raw']}"
    )

    selected_name, feature_cols, feat_ablation = select_feature_preset(
        train, valid, macro_source, fund_included
    )
    write_ablation_report(
        reports / "feature_ablation.json",
        selected_name,
        feature_cols,
        macro_source,
        fund_included,
        feat_ablation,
    )

    feature_cols, train, valid, test, train_ready, used_stock_freq, stock_freq_info = (
        _apply_stock_freq(train, valid, test, train_ready, feature_cols)
    )
    stock_freq_map = stock_freq_info.get("freq_map") or {}

    dir_model, dir_sel, model_ablation = select_direction_head(
        train, valid, feature_cols=feature_cols
    )
    ablation_path = reports / "model_ablation.json"
    ablation_path.write_text(json.dumps(model_ablation, indent=2, default=str), encoding="utf-8")
    print(
        f"model_ablation: selected={model_ablation.get('selected')} "
        f"ic={model_ablation.get('selected_rank_ic')}"
    )

    results: dict = {
        "problem": {
            "direction": "5d_excess_vs_NASI",
            "dead_zone_xs": DEAD_ZONE_XS,
            "return_target": "y_ret_5d_xs",
        },
        "protocol": protocol,
        "universe": universe,
        "investable": investable,
        "macro_source": macro_source,
        "feature_cache": cache_info,
        "fundamentals_status": fund_status,
        "feature_ablation": {"selected": selected_name, "n_features": len(feature_cols)},
        "model_ablation": {"selected": model_ablation.get("selected"), "path": str(ablation_path)},
        "model_features": feature_cols,
        "stock_freq_feature": used_stock_freq,
        "stock_freq_info": {k: v for k, v in stock_freq_info.items() if k != "freq_map"},
        "n_rows": len(train_ready),
        "splits": {
            "train": split_meta(train),
            "valid": split_meta(valid),
            "test": split_meta(test),
        },
        "selection": dir_sel,
        "baselines": {},
        "models": {},
        "diagnostics": {},
        "walk_forward": [],
        "stress": {},
        "classical": {},
    }

    eval_df = test if len(test) else (valid if len(valid) else train_ready.tail(500))
    if eval_df.empty:
        print("No evaluation rows", file=sys.stderr)
        return 1

    results["baselines"] = _baselines(eval_df)
    dir_model.save(settings.models_dir / "direction_model.joblib")
    dir_test, dir_pred = direction_metrics_bundle(eval_df, dir_model)
    results["models"]["lightgbm_direction"] = dir_test
    results["models"]["direction_train_only_valid"] = dir_sel.get("valid_direction") or {}
    results["diagnostics"].update(direction_diagnostics(eval_df, dir_pred))

    if len(valid) and valid["y_dir_5d"].notna().sum() >= 50:
        vpred = dir_model.predict(valid)
        yp_f = binary_from_labels(vpred["pred_label"])
        mask = ~np.isnan(yp_f)
        if mask.sum() >= 20:
            pack = calibration_pack(
                valid.loc[mask, "y_dir_5d"].astype(int).values,
                vpred.loc[mask, "prob_up"].values,
                pred_label=vpred["pred_label"],
            )
            results["diagnostics"]["calibration_valid"] = {
                k: v for k, v in pack.items() if k != "calibration_bins"
            }

    err_hist = build_error_history(eval_df, dir_pred, horizon_days=LABEL_HORIZON_DAYS)
    err_path = settings.data_processed_dir / "error_history.parquet"
    err_hist.to_parquet(err_path, index=False)
    results["error_history_path"] = str(err_path)
    results["error_history_rows"] = int(len(err_hist))

    if hasattr(dir_model, "shap_summary"):
        shap_df = dir_model.shap_summary(eval_df)
        if shap_df is not None:
            shap_df.to_csv(reports / "shap_direction.csv", index=False)
            results["models"]["shap_top5"] = shap_df.head(5).to_dict(orient="records")

    ret_model, ret_sel = fit_return_selected(
        train, valid, feature_cols=feature_cols, target_col="y_ret_5d_xs"
    )
    results["selection"].update(ret_sel)
    ret_bits = eval_return_model(eval_df, ret_model)
    results["models"].update({k: v for k, v in ret_bits.items() if k != "rank_ic_return_model"})
    if "rank_ic_return_model" in ret_bits:
        results["diagnostics"]["rank_ic_return_model"] = ret_bits["rank_ic_return_model"]
    ret_model.save(settings.models_dir / "return_model.joblib")

    try:
        raw_ret, _ = fit_return_selected(
            train, valid, feature_cols=feature_cols, target_col="y_ret_5d", tune=False
        )
        raw_ev = eval_df.dropna(subset=["y_ret_5d"])
        raw_pred = raw_ret.predict(raw_ev)
        results["models"]["lightgbm_return_5d_raw_diagnostic"] = regression_metrics(
            raw_ev["y_ret_5d"].values, raw_pred["pred_value"].values
        )
    except Exception as exc:  # noqa: BLE001
        results["models"]["lightgbm_return_5d_raw_diagnostic"] = {"error": str(exc)}

    fit_final = pd.concat([train, valid], ignore_index=True) if len(valid) else train
    vol_model = VolatilityModel(feature_cols=feature_cols).fit(
        fit_final.dropna(subset=["y_vol_20d"])
    )
    vol_eval = eval_df.dropna(subset=["y_vol_20d"])
    if len(vol_eval):
        vol_pred = vol_model.predict(vol_eval)
        results["models"]["lightgbm_vol_20d"] = regression_metrics(
            vol_eval["y_vol_20d"].values, vol_pred["pred_value"].values
        )
    vol_model.save(settings.models_dir / "volatility_model.joblib")

    per_stock = per_stock_direction(eval_df, dir_pred)
    results["diagnostics"]["per_stock"] = per_stock
    flagged = [r["stock_code"] for r in per_stock if r.get("flag")]
    results["diagnostics"]["flagged_stocks"] = flagged
    flagged_frac = len(flagged) / max(len(per_stock), 1)
    pd.DataFrame(per_stock).to_csv(reports / "per_stock_direction.csv", index=False)
    results["diagnostics"]["rank_ic"] = rank_ic(
        eval_df.merge(
            dir_pred[["trade_date", "stock_code", "prob_up"]],
            on=["trade_date", "stock_code"],
            how="inner",
        ),
        "prob_up",
        "y_ret_5d_xs",
    )

    stress_ric, stress_acc = stress_oos_retrain(train_ready, feature_cols, dir_sel)
    results["stress_rank_ic"] = stress_ric
    results["stress"] = stress_acc
    fragile = is_stress_fragile(stress_ric)
    results["stress_fragile"] = fragile

    wf = run_walk_forward_direction(
        train_ready,
        fallback_threshold=float(dir_sel.get("threshold") or 0.5),
        feature_cols=feature_cols,
        params=dir_sel.get("best_params") or {},
        decision_mode=str(dir_sel.get("decision_mode") or "threshold"),
        direction_head=str(dir_sel.get("direction_head") or "binary"),
        horizon_days=LABEL_HORIZON_DAYS,
    )
    results["walk_forward"] = wf
    wf_sum = walk_forward_summary(wf)
    results["walk_forward_summary"] = wf_sum

    cov = (results["models"].get("return_interval_coverage") or {}).get("coverage")
    target_cov = float((ret_sel.get("return_quantiles") or {}).get("target_coverage") or 0.80)
    gates = compute_validation_gates(
        test_rank_ic=(results["diagnostics"].get("rank_ic") or {}).get("mean_ic"),
        wf_mean_rank_ic=wf_sum.get("wf_mean_rank_ic"),
        stress_fragile=fragile,
        flagged_frac=flagged_frac,
        return_coverage=cov,
        target_coverage=target_cov,
        macro_source=macro_source,
    )
    results["validation_gates"] = gates
    print(f"validation_status={gates.get('status')} reasons={gates.get('reasons')}")

    try:
        px_test = prices[
            prices["stock_code"].isin(eval_df["stock_code"].unique())
            & prices["trade_date"].isin(eval_df["trade_date"].unique())
        ]
        results["research_backtest"] = simulate_book_long_flat(px_test, dir_pred, cost_bps=50.0)
    except Exception as exc:  # noqa: BLE001
        results["research_backtest"] = {
            "ok": False,
            "error": str(exc),
            "disclaimer": "research_only_not_for_selection",
        }

    sample_code = investable[0] if investable else universe[0]
    hist = prices[prices["stock_code"] == sample_code].sort_values("trade_date")
    results["classical"]["arima_example"] = {
        "stock": sample_code,
        **fit_arima_forecast(hist["price"], steps=5),
    }
    results["classical"]["garch_example"] = {
        "stock": sample_code,
        **fit_garch_vol(hist["price"].pct_change(), horizon=20),
    }

    write_model_meta(
        feature_cols=feature_cols,
        macro_source=macro_source,
        direction_head=str(dir_sel.get("direction_head") or "binary"),
        threshold=float(dir_sel.get("threshold") or 0.5),
        abstain_band=float(dir_sel.get("abstain_band") or 0.05),
        calibration_status=str(dir_sel.get("calibration_status") or "none"),
        decision_mode=str(dir_sel.get("decision_mode") or "threshold"),
        best_params=dir_sel.get("best_params") or {},
        target_return="y_ret_5d_xs",
        extra={
            "stock_freq_feature": used_stock_freq,
            "stock_freq_map": stock_freq_map if used_stock_freq else {},
            "best_iteration": dir_sel.get("best_iteration"),
            "protocol_embargo": True,
            "label_horizon_days": LABEL_HORIZON_DAYS,
        },
    )

    live = score_latest_forecasts()
    results["latest_forecasts_path"] = live.get("path")
    results["latest_score"] = live

    cm = results["diagnostics"].get("confusion") or {}
    if cm.get("tn", 1) == 0 and (dir_test.get("recall_up") or 0) >= 0.99:
        if dir_sel.get("decision_mode") != "ranked":
            print(
                "WARNING: hard labels look always-up; check decision_mode/calibration",
                file=sys.stderr,
            )
        results["always_up_collapse_warning"] = True

    report_path = reports / "train_val_report.md"
    write_train_val_report(results, report_path)
    results["train_val_report"] = str(report_path)

    val_path = reports / "validation_report.json"
    write_validation_report(results, val_path)
    results["validation_report"] = str(val_path)

    out_json = reports / "evaluation_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    summary = {
        "protocol": {
            k: protocol.get(k) for k in ("embargo", "horizon_days", "train_end", "valid_end")
        },
        "validation_gates": gates,
        "selection": {
            k: results["selection"].get(k)
            for k in (
                "direction_head",
                "decision_mode",
                "threshold",
                "calibration_status",
                "best_params",
                "valid_rank_ic",
            )
        },
        "calibration": results["diagnostics"].get("calibration"),
        "walk_forward_summary": wf_sum,
        "rank_ic": results["diagnostics"].get("rank_ic"),
        "stress_fragile": fragile,
        "research_backtest": {
            k: results["research_backtest"].get(k)
            for k in ("ok", "total_return", "max_drawdown", "n_names", "disclaimer")
            if isinstance(results.get("research_backtest"), dict)
        },
        "latest_score": live,
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_json}")
    print(f"wrote {report_path}")
    print(f"wrote {val_path}")
    print(f"wrote {ablation_path}")
    print(f"wrote {settings.models_dir / 'model_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
