"use client";

import Link from "next/link";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Forecast } from "../lib/api";
import { callLabel, pct, rangeLabel, stockPath } from "../lib/format";
import { CompanyLogo } from "./CompanyLogo";
import { StockLink } from "./StockLink";

type Props = {
  rows: Forecast[];
  favourites?: string[];
  onToggleFavourite?: (code: string) => void;
  onTicket?: (row: Forecast) => void;
  busy?: string | null;
};

export function ForecastBoard({
  rows,
  favourites = [],
  onToggleFavourite,
  onTicket,
  busy,
}: Props) {
  const favSet = new Set(favourites.map((c) => c.toUpperCase()));
  const sorted = [...rows].sort((a, b) => Number(b.prob_up || 0) - Number(a.prob_up || 0));
  const chartData = sorted
    .filter((s) => s.prob_up != null)
    .slice(0, 15)
    .map((s) => ({ code: s.stock_code, prob: Number(s.prob_up) }));

  if (!sorted.length) {
    return <p className="muted">No names match the current filters.</p>;
  }

  return (
    <div className="stack-2">
      <div className="panel">
        <table className="data-table">
          <thead>
            <tr>
              <th />
              <th>Stock</th>
              <th>Price</th>
              <th>Call</th>
              <th>P(up)</th>
              <th>Range</th>
              <th>Vol</th>
              <th>Hit</th>
              <th>Book</th>
              <th>Suggest</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const code = r.stock_code.toUpperCase();
              const isFav = favSet.has(code);
              const canTicket =
                !!onTicket && !!r.suggested_side && (r.suggested_shares || 0) > 0;
              return (
                <tr key={code}>
                  <td>
                    {onToggleFavourite && (
                      <button
                        type="button"
                        className={`fav-btn${isFav ? " on" : ""}`}
                        aria-label={isFav ? "Unfavourite" : "Favourite"}
                        onClick={() => onToggleFavourite(code)}
                      >
                        {isFav ? "★" : "☆"}
                      </button>
                    )}
                  </td>
                  <td>
                    <StockLink code={code} className="ticker-cell">
                      <CompanyLogo code={code} />
                      <strong>{code}</strong>
                    </StockLink>
                  </td>
                  <td>{r.price != null ? Number(r.price).toFixed(2) : "—"}</td>
                  <td>{callLabel(r.direction_5d, r.pred_label)}</td>
                  <td>{pct(r.prob_up, 0)}</td>
                  <td>{rangeLabel(r.ret5d_low, r.ret5d_high)}</td>
                  <td>{pct(r.vol20d, 1)}</td>
                  <td>{pct(r.hit_rate, 0)}</td>
                  <td>
                    <span className="pill">{r.book_status || "flat"}</span>
                  </td>
                  <td>
                    {canTicket
                      ? `${String(r.suggested_side).toUpperCase()} ${r.suggested_shares}`
                      : r.suggest_note || "—"}
                  </td>
                  <td>
                    <div className="btn-row">
                      <Link className="btn" href={stockPath(code)}>
                        Open
                      </Link>
                      {canTicket && (
                        <button
                          className="btn btn-primary"
                          disabled={busy === `${code}:ticket`}
                          onClick={() => onTicket?.(r)}
                          title="Pending paper ticket — not a live order"
                        >
                          Ticket
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="panel">
        <p className="section-label" style={{ marginTop: 0 }}>
          P(up) board
        </p>
        <div className="chart-wrap">
          <ResponsiveContainer>
            <BarChart data={chartData}>
              <XAxis dataKey="code" tick={{ fontSize: 11, fill: "#5c6b7a" }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#5c6b7a" }} />
              <Tooltip formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "P(up)"]} />
              <Bar dataKey="prob" fill="#2f7d4a" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
