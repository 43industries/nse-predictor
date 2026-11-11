"""Stretch LSTM direction head — kept only if it beats LightGBM on the same test panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def fit_predict_lstm_direction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    seq_len: int = 20,
    epochs: int = 8,
    hidden: int = 32,
) -> dict:
    """Train a small LSTM on sequences; return metrics + whether it beats always-up/LGBM placeholders.

    Returns ok=False with error if torch missing or data too thin.
    """
    if not _torch_available():
        return {"ok": False, "skipped": True, "error": "torch_unavailable"}

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        cols = [c for c in feature_cols if c in df.columns][:24]  # keep LSTM light
        d = df.dropna(subset=["y_dir_5d"]).copy()
        if len(d) < seq_len + 50 or not cols:
            return None
        d = d.sort_values(["stock_code", "trade_date"])
        xs, ys = [], []
        for _, g in d.groupby("stock_code"):
            if len(g) <= seq_len + 5:
                continue
            X = g[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            y = g["y_dir_5d"].astype(int).to_numpy()
            for i in range(seq_len, len(g)):
                xs.append(X[i - seq_len : i])
                ys.append(y[i])
        if len(xs) < 80:
            return None
        return np.stack(xs), np.asarray(ys, dtype=np.int64)

    tr = _xy(train)
    va = _xy(valid) if len(valid) else None
    te = _xy(test)
    if tr is None or te is None:
        return {"ok": False, "error": "insufficient_sequences"}

    Xtr, ytr = tr
    Xte, yte = te
    device = torch.device("cpu")

    class _LSTM(nn.Module):
        def __init__(self, n_feat: int):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = _LSTM(Xtr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=256,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == yte).mean())
    always_up = float((yte == 1).mean())
    beats_always_up = acc > always_up + 0.005
    return {
        "ok": True,
        "method": "lstm",
        "accuracy": acc,
        "always_up_rate": always_up,
        "beats_always_up": beats_always_up,
        "n_test": int(len(yte)),
        "seq_len": seq_len,
        "epochs": epochs,
        "keep_model": False,  # caller compares to LightGBM
    }


def maybe_save_lstm(payload: dict, path: Path) -> None:
    if not payload.get("ok") or not payload.get("keep_model"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(payload), encoding="utf-8")
