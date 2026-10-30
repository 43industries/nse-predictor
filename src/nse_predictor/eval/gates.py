"""Pass / warn / fail validation gates (do not auto-block deploy)."""

from __future__ import annotations

import math


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def compute_validation_gates(
    *,
    test_rank_ic: float | None,
    wf_mean_rank_ic: float | None,
    stress_fragile: bool,
    flagged_frac: float,
    return_coverage: float | None = None,
    target_coverage: float = 0.80,
    macro_source: str | None = None,
    flagged_frac_warn: float = 0.40,
    coverage_slack: float = 0.10,
) -> dict:
    """Return status pass|warn|fail with reasons.

    Fail only if test Rank IC ≤ 0 and WF mean Rank IC ≤ 0.
    Warn on fragile stress, high per-stock flag rate, coverage shortfall, synthetic macro.
    """
    reasons: list[str] = []
    flags: list[str] = []
    tric = float(test_rank_ic) if _finite(test_rank_ic) else float("nan")
    wic = float(wf_mean_rank_ic) if _finite(wf_mean_rank_ic) else float("nan")

    fail = _finite(tric) and tric <= 0 and _finite(wic) and wic <= 0
    if fail:
        reasons.append(f"test_rank_ic={tric:.3f} and wf_mean_rank_ic={wic:.3f} both ≤ 0")
        flags.append("rank_ic_fail")
    if stress_fragile:
        reasons.append("stress_fragile: at least one stress Rank IC < 0")
        flags.append("stress_fragile")
    if flagged_frac >= flagged_frac_warn:
        reasons.append(f"flagged_frac={flagged_frac:.0%} ≥ {flagged_frac_warn:.0%}")
        flags.append("per_stock_flags")
    if _finite(return_coverage) and float(return_coverage) < (target_coverage - coverage_slack):
        reasons.append(
            f"return_coverage={float(return_coverage):.1%} < "
            f"target {target_coverage:.0%}−{coverage_slack:.0%}"
        )
        flags.append("coverage_shortfall")
    if macro_source and str(macro_source).startswith("synthetic"):
        reasons.append(f"macro_source={macro_source}")
        flags.append("synthetic_macro")

    status = "fail" if fail else ("warn" if reasons else "pass")
    return {
        "status": status,
        "reasons": reasons,
        "flags": flags,
        "test_rank_ic": tric if _finite(tric) else None,
        "wf_mean_rank_ic": wic if _finite(wic) else None,
        "stress_fragile": bool(stress_fragile),
        "flagged_frac": float(flagged_frac),
        "return_coverage": float(return_coverage) if _finite(return_coverage) else None,
        "macro_source": macro_source,
    }
