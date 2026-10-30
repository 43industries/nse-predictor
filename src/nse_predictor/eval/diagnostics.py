"""Train/validation diagnostics for direction and return models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from nse_predictor.eval.metrics import (
    abstain_rate,
    brier_score,
    direction_metrics,
    expected_calibration_error,
)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": i,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0,
                    "mean_prob": None,
                    "frac_up": None,
                }
            )
            continue
        rows.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_prob": float(y_prob[mask].mean()),
                "frac_up": float(y_true[mask].mean()),
            }
        )
    return rows


def binary_from_labels(pred_label: pd.Series | np.ndarray) -> np.ndarray:
    """Map up/down/flat → 1/0/nan (flat excluded from hard metrics)."""
    s = pd.Series(pred_label).astype(str).str.lower()
    out = np.full(len(s), np.nan)
    out[s == "up"] = 1.0
    out[s == "down"] = 0.0
    return out


def per_stock_direction(
    eval_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    min_n: int = 80,
) -> list[dict]:
    merged = eval_df[["trade_date", "stock_code", "y_dir_5d"]].merge(
        pred_df[["trade_date", "stock_code", "prob_up", "pred_label"]],
        on=["trade_date", "stock_code"],
        how="inner",
    )
    rows: list[dict] = []
    for code, g in merged.groupby("stock_code"):
        y = g["y_dir_5d"].astype(int).values
        yp_f = binary_from_labels(g["pred_label"])
        mask = ~np.isnan(yp_f)
        if mask.sum() < 10:
            rows.append(
                {
                    "stock_code": str(code),
                    "accuracy": float("nan"),
                    "n": int(len(g)),
                    "n_scored": int(mask.sum()),
                    "flag": "thin_n",
                }
            )
            continue
        yp = yp_f[mask].astype(int)
        yt = y[mask]
        m = direction_metrics(yt, yp, g.loc[mask, "prob_up"].values)
        base = float(m.get("base_rate_up") or 0.5)
        chance = max(base, 1.0 - base)
        flag = ""
        if m["n"] < min_n:
            flag = "thin_n"
        elif m["accuracy"] <= chance + 0.01:
            flag = "at_chance"
        rows.append({"stock_code": str(code), **m, "flag": flag})
    rows.sort(key=lambda r: (r.get("accuracy") is not None, r.get("accuracy") or 0), reverse=True)
    return rows


def _is_always_up(y_true: np.ndarray, pred: np.ndarray) -> bool:
    cm = confusion_counts(y_true, pred)
    recall = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    return cm["tn"] == 0 or recall >= 0.99


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
    min_threshold: float = 0.35,
    max_threshold: float = 0.65,
    metric: str = "f1",
    reject_always_up: bool = True,
) -> dict:
    """Pick threshold on F1 or balanced accuracy within bounds; reject always-up."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    lo, hi = float(min_threshold), float(max_threshold)
    if grid is None:
        base = float(y_true.mean()) if len(y_true) else 0.5
        grid = np.unique(
            np.concatenate(
                [
                    np.linspace(lo, hi, 31),
                    np.linspace(max(lo, base - 0.15), min(hi, base + 0.15), 15),
                ]
            )
        )
    grid = np.asarray([t for t in grid if lo - 1e-9 <= float(t) <= hi + 1e-9], dtype=float)
    if len(grid) == 0:
        grid = np.array([0.5])

    best_t = 0.5
    best_score = -1.0
    best_metrics: dict = {}
    rejected = 0
    for t in grid:
        pred = (y_prob >= float(t)).astype(int)
        if reject_always_up and _is_always_up(y_true, pred):
            rejected += 1
            continue
        f1 = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
        bal = float(balanced_accuracy_score(y_true, pred))
        m = direction_metrics(y_true, pred, y_prob)
        score = f1 if metric == "f1" else bal
        if score > best_score + 1e-12:
            best_score = score
            best_t = float(t)
            best_metrics = {**m, "f1_up": f1, "balanced_acc": bal}

    always_up_acc = float(y_true.mean()) if len(y_true) else 0.5
    found = bool(best_metrics)
    beats = found and (best_metrics.get("accuracy", 0) > always_up_acc + 0.005)
    return {
        "threshold": best_t if found else 0.5,
        "valid_metrics": best_metrics,
        "f1_up": best_metrics.get("f1_up", 0.0),
        "found": found,
        "beats_always_up": beats,
        "rejected_always_up": rejected,
        "metric": metric,
        "min_threshold": lo,
        "max_threshold": hi,
    }


def calibration_pack(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pred_label: pd.Series | np.ndarray | None = None,
    n_bins: int = 10,
) -> dict:
    """Brier, ECE, prob summary, bins; optional abstain_rate from labels."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    out: dict = {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "mean_prob_up": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
        "std_prob_up": float(np.std(y_prob)) if len(y_prob) else float("nan"),
        "n_unique_probs": int(pd.Series(np.round(y_prob, 4)).nunique()) if len(y_prob) else 0,
        "n": int(len(y_true)),
        "calibration_bins": calibration_table(y_true, y_prob, n_bins=n_bins),
    }
    if pred_label is not None:
        out["abstain_rate"] = abstain_rate(pred_label)
        out["n_flat"] = int((pd.Series(pred_label).astype(str).str.lower() == "flat").sum())
    return out


def write_train_val_report(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Train / validation report")
    lines.append("")
    lines.append("Decision-support research only - not investment advice.")
    lines.append("")
    lines.append("## Problem definition")
    lines.append(
        "- **Direction:** P(5d excess vs NASI **> +1%**); label **down** if excess **< -1%**; "
        "rows with |excess| <= 1% are excluded (dead zone)."
    )
    lines.append("- **Return model:** 5d **excess** vs NASI + residual quantile band.")
    lines.append("- **Universe:** investable ordinary shares (liquidity / price / history gates).")
    lines.append("")
    if results.get("investable"):
        lines.append(f"- Investable names: {', '.join(results['investable'])}")
        lines.append("")
    gates = results.get("validation_gates") or {}
    if gates.get("status"):
        lines.append(
            f"> **validation_status={gates.get('status')}**"
            + (f" — {'; '.join(gates.get('reasons') or [])}" if gates.get("reasons") else "")
        )
        lines.append("")
    if results.get("stress_fragile"):
        lines.append("> **stress_fragile=true** — stress-window Rank IC < 0 on at least one slice.")
        lines.append("")
    proto = results.get("protocol") or {}
    if proto:
        lines.append("## Protocol")
        lines.append(
            f"- Embargo: **{proto.get('embargo')}** · horizon **{proto.get('horizon_days')}d** · "
            f"train_end `{proto.get('train_end')}` · valid_end `{proto.get('valid_end')}`"
        )
        if proto.get("n_train_fit") is not None:
            lines.append(
                f"- Fit rows after embargo: train {proto.get('n_train_fit')} "
                f"(raw {proto.get('n_train_raw')}), valid {proto.get('n_valid_fit')} "
                f"(raw {proto.get('n_valid_raw')})"
            )
        lines.append("")
    splits = results.get("splits", {})
    lines.append("## Time splits")
    for name, vals in splits.items():
        lines.append(f"- **{name}**: {vals}")
    lines.append("")
    sel = results.get("selection", {})
    if sel:
        lines.append("## Selection (from validation)")
        lines.append(f"- Direction head: **{sel.get('direction_head', 'binary')}**")
        lines.append(f"- Decision mode: **{sel.get('decision_mode', 'threshold')}**")
        lines.append(f"- Direction threshold: **{sel.get('threshold', 0.5):.2f}**")
        lines.append(f"- Calibration: **{sel.get('calibration', 'none')}** ({sel.get('calibration_status', '')})")
        if sel.get("best_params"):
            lines.append(f"- Best params: `{sel.get('best_params')}`")
        vm = sel.get("valid_direction") or {}
        if vm:
            lines.append(
                f"- Valid accuracy: **{vm.get('accuracy', 0):.1%}** "
                f"(lift vs always-up {vm.get('lift_vs_always_up', 0):+.1%})"
            )
        rq = sel.get("return_quantiles") or {}
        if rq:
            lines.append(
                f"- Return residual quantiles (excess): "
                f"[{rq.get('q_low', 0):.3f}, {rq.get('q_high', 0):.3f}] "
                f"(target ~{rq.get('target_coverage', 0.8):.0%} coverage)"
            )
        lines.append("")
    bas = results.get("baselines", {})
    mod = results.get("models", {})
    lines.append("## Test holdout (direction on excess labels)")
    for key, label in [
        ("always_up", "Always up"),
        ("majority", "Majority class"),
        ("ma_5_20", "MA 5/20"),
    ]:
        m = bas.get(key) or {}
        if m:
            lines.append(f"- {label}: {m.get('accuracy', 0):.1%}")
    dm = mod.get("lightgbm_direction") or {}
    if dm:
        lines.append(
            f"- LightGBM (selected): **{dm.get('accuracy', 0):.1%}** "
            f"(precision_up {dm.get('precision_up', 0):.1%}, "
            f"recall_up {dm.get('recall_up', 0):.1%}, "
            f"n_scored {dm.get('n', 0)}, "
            f"abstain {dm.get('abstain_rate', dm.get('n_flat', 0))})"
        )
        if dm.get("abstain_rate") is not None:
            lines.append(f"- Abstain rate (flat): **{dm.get('abstain_rate', 0):.1%}**")
    ric = results.get("diagnostics", {}).get("rank_ic") or {}
    if ric.get("n_days"):
        lines.append(
            f"- Rank IC (prob_up vs y_ret_5d_xs): **mean {ric.get('mean_ic', float('nan')):.3f}** "
            f"(median {ric.get('median_ic', float('nan')):.3f}, days {ric.get('n_days')})"
        )
    lines.append("")
    cal = results.get("diagnostics", {}).get("calibration") or {}
    if cal:
        lines.append("## Calibration (test, non-flat)")
        lines.append(
            f"- Brier **{cal.get('brier', float('nan')):.4f}** · "
            f"ECE **{cal.get('ece', float('nan')):.4f}** · "
            f"std(prob) **{cal.get('std_prob_up', float('nan')):.3f}** · "
            f"unique probs **{cal.get('n_unique_probs', 0)}**"
        )
        lines.append("")
    cm = results.get("diagnostics", {}).get("confusion") or {}
    if cm:
        lines.append("## Confusion (test, non-flat only)")
        lines.append(
            f"- TP {cm.get('tp')} · FP {cm.get('fp')} · TN {cm.get('tn')} · FN {cm.get('fn')}"
        )
        lines.append("")
    cov = mod.get("return_interval_coverage") or {}
    if cov:
        lines.append("## Excess return interval (test)")
        lines.append(
            f"- Coverage **{cov.get('coverage', 0):.1%}**, "
            f"mean width **{cov.get('mean_width', 0):.1%}**"
        )
        lines.append("")
    stress = results.get("stress_rank_ic") or {}
    if stress:
        lines.append("## Stress Rank IC (OOS retrain)")
        for k, v in stress.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    per = results.get("diagnostics", {}).get("per_stock") or []
    if per:
        lines.append("## Per-stock test accuracy (top / bottom)")
        lines.append("| Stock | Accuracy | N | Flag |")
        lines.append("| --- | --- | --- | --- |")
        show = list(per[:5])
        if len(per) > 10:
            show += [{"stock_code": "…"}] + per[-5:]
        elif len(per) > 5:
            show = per
        for r in show:
            if r.get("stock_code") == "…":
                lines.append("| … | | | |")
                continue
            flag = r.get("flag") or ""
            acc = r.get("accuracy")
            acc_s = f"{acc:.1%}" if isinstance(acc, float) and acc == acc else "—"
            lines.append(f"| {r.get('stock_code')} | {acc_s} | {r.get('n', 0)} | {flag} |")
        flagged = [r for r in per if r.get("flag")]
        if flagged:
            lines.append("")
            lines.append(
                f"- Flagged names: {', '.join(r['stock_code'] for r in flagged)}"
            )
        lines.append("")
    wf = results.get("walk_forward") or []
    wf_sum = results.get("walk_forward_summary") or {}
    if wf:
        beats = sum(1 for f in wf if f.get("beats_baseline"))
        lines.append("## Walk-forward")
        if wf_sum:
            lines.append(
                f"- Mean Rank IC: **{wf_sum.get('wf_mean_rank_ic', float('nan')):.3f}** · "
                f"IC>0 folds: **{wf_sum.get('wf_folds_ic_positive', 0)}/{len(wf)}** · "
                f"beats always-up: **{wf_sum.get('wf_folds_beat_always_up', beats)}/{len(wf)}**"
            )
        else:
            lines.append(f"- Folds beating always-up: **{beats}/{len(wf)}**")
        for f in wf:
            m = f.get("model") or {}
            ric_f = f.get("rank_ic") or {}
            ic = ric_f.get("mean_ic", float("nan"))
            lines.append(
                f"- {f.get('fold')}: acc {m.get('accuracy', 0):.1%} · "
                f"Rank IC {'—' if ic != ic else f'{ic:.3f}'} · "
                f"abstain {f.get('abstain_rate', 0):.1%} · "
                f"beats={f.get('beats_baseline')}"
            )
        lines.append("")
    bt = results.get("research_backtest") or {}
    if bt:
        lines.append("## Research backtest (not for selection)")
        lines.append(
            f"- Equal-weight long/flat · total return **{bt.get('total_return', float('nan')):.1%}** · "
            f"buy-hold **{bt.get('buy_hold_return', float('nan')):.1%}** · "
            f"max DD **{bt.get('max_drawdown', float('nan')):.1%}** · "
            f"names {bt.get('n_names', 0)}"
        )
        lines.append(f"- Disclaimer: `{bt.get('disclaimer', 'research_only')}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
