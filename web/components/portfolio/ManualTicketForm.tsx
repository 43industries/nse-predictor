"use client";

import { useState } from "react";

type Props = {
  lotSize?: number;
  busy?: boolean;
  onSubmit: (body: {
    stock_code: string;
    side: "buy" | "sell";
    shares: number;
    reason?: string;
  }) => void;
};

export function ManualTicketForm({ lotSize = 100, busy, onSubmit }: Props) {
  const [code, setCode] = useState("");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [shares, setShares] = useState(lotSize);
  const [reason, setReason] = useState("");

  return (
    <form
      className="panel form-grid"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit({
          stock_code: code.trim().toUpperCase(),
          side,
          shares: Number(shares),
          reason: reason.trim() || "manual",
        });
      }}
    >
      <p className="section-label" style={{ marginTop: 0, gridColumn: "1 / -1" }}>
        Manual ticket
      </p>
      <label>
        Side
        <select value={side} onChange={(e) => setSide(e.target.value as "buy" | "sell")}>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
      </label>
      <label>
        Code
        <input value={code} onChange={(e) => setCode(e.target.value)} required placeholder="EQTY" />
      </label>
      <label>
        Shares <span className="muted">(lot {lotSize})</span>
        <input
          type="number"
          min={1}
          step={lotSize}
          value={shares}
          onChange={(e) => setShares(Number(e.target.value))}
          required
        />
      </label>
      <label>
        Reason
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="optional" />
      </label>
      <button className="btn btn-primary" type="submit" disabled={busy}>
        {busy ? "…" : "Create ticket"}
      </button>
    </form>
  );
}
