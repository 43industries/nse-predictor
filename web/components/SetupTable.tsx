"use client";

import Link from "next/link";
import type { SetupRow } from "../lib/api";
import { callLabel, pct, rangeLabel, stockPath } from "../lib/format";
import { CompanyLogo } from "./CompanyLogo";
import { StockLink } from "./StockLink";

type Props = {
  rows: SetupRow[];
  onLog?: (row: SetupRow, stance: "long" | "skip") => void;
  busy?: string | null;
};

export function SetupTable({ rows, onLog, busy }: Props) {
  if (!rows.length) {
    return <p className="muted">No actionable up-setups after excluding at-chance names.</p>;
  }
  return (
    <div className="panel">
      <table className="data-table">
        <thead>
          <tr>
            <th>Stock</th>
            <th>Price</th>
            <th>Call</th>
            <th>P(up)</th>
            <th>Range</th>
            <th>Hit</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.stock_code}>
              <td>
                <StockLink code={r.stock_code} className="ticker-cell">
                  <CompanyLogo code={r.stock_code} />
                  <strong>{r.stock_code}</strong>
                </StockLink>
              </td>
              <td>{r.price != null ? Number(r.price).toFixed(2) : "—"}</td>
              <td>{callLabel(r.pred_label)}</td>
              <td>{pct(r.prob_up, 0)}</td>
              <td>{rangeLabel(r.ret5d_lo, r.ret5d_hi)}</td>
              <td>{pct(r.hit_rate, 0)}</td>
              <td>
                <div className="btn-row">
                  <Link className="btn" href={stockPath(r.stock_code)}>
                    Open
                  </Link>
                  {onLog && (
                    <>
                      <button
                        className="btn btn-accent"
                        disabled={busy === `${r.stock_code}:long`}
                        onClick={() => onLog(r, "long")}
                      >
                        Log long
                      </button>
                      <button
                        className="btn"
                        disabled={busy === `${r.stock_code}:skip`}
                        onClick={() => onLog(r, "skip")}
                      >
                        Log skip
                      </button>
                    </>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
