"use client";

import { useState } from "react";

type Props = {
  busy?: boolean;
  onSubmit: (body: { stock_code: string; shares: number; avg_cost: number }) => void;
};

export function SeedPositionForm({ busy, onSubmit }: Props) {
  const [code, setCode] = useState("");
  const [shares, setShares] = useState(100);
  const [avgCost, setAvgCost] = useState(10);

  return (
    <form
      className="panel form-grid"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit({
          stock_code: code.trim().toUpperCase(),
          shares: Number(shares),
          avg_cost: Number(avgCost),
        });
      }}
    >
      <p className="section-label" style={{ marginTop: 0, gridColumn: "1 / -1" }}>
        Seed holdings
      </p>
      <label>
        Code
        <input value={code} onChange={(e) => setCode(e.target.value)} required placeholder="SCOM" />
      </label>
      <label>
        Shares
        <input
          type="number"
          min={0}
          step={1}
          value={shares}
          onChange={(e) => setShares(Number(e.target.value))}
          required
        />
      </label>
      <label>
        Avg cost
        <input
          type="number"
          min={0.01}
          step={0.01}
          value={avgCost}
          onChange={(e) => setAvgCost(Number(e.target.value))}
          required
        />
      </label>
      <button className="btn btn-accent" type="submit" disabled={busy}>
        {busy ? "…" : "Seed / replace"}
      </button>
      <p className="muted small" style={{ gridColumn: "1 / -1" }}>
        Ops seed — sets position without cash accounting. Use 0 shares to clear a name.
      </p>
    </form>
  );
}
