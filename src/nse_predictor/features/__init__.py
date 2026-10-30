from .build import DEAD_ZONE_XS, build_feature_matrix
from .registry import available_features, model_feature_cols

__all__ = [
    "DEAD_ZONE_XS",
    "build_feature_matrix",
    "available_features",
    "model_feature_cols",
    "select_feature_preset",
    "write_ablation_report",
]


def __getattr__(name: str):
    if name in ("select_feature_preset", "write_ablation_report"):
        from .ablation import select_feature_preset, write_ablation_report

        return {
            "select_feature_preset": select_feature_preset,
            "write_ablation_report": write_ablation_report,
        }[name]
    raise AttributeError(name)
