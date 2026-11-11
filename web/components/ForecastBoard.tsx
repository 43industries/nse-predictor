"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Forecast } from "../lib/api";
import { callLabel, pct, rangeLabel } from "../lib/format";
import { CompanyLogo } from "./CompanyLogo";
import { StockLink } from "./StockLink";

export function ForecastBoard({ rows }: { rows: Forecast[] }) {
  const sorted = [...rows].sort((a, b) => Number(b.prob_up || 0) - Number(a.prob_up || 0));
  const chartData = sorted
    .filter((s) => s.prob_up != null)
    .slice(0, 15)
    .map((s) => ({ code: s.stock_code, prob: Number(s.prob_up) }));

  return (
    <div className="stack-2">
      <div className="panel">
        <table className="data-table">
          <thead>
            <tr>
              <th>Stock</th>
              <th>Price</th>
              <th>Call</th>
              <th>P(up)</th>
              <th>Range</th>
              <th>Vol</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.stock_code}>
                <td>
                  <StockLink code={r.stock_code} className="ticker-cell">
                    <CompanyLogo code={r.stock_code} />
                    <strong>{r.stock_code}</strong>
                  </StockLink>
                </td>
                <td>{r.price != null ? Number(r.price).toFixed(2) : "—"}</td>
                <td>{callLabel(r.direction_5d, r.pred_label)}</td>
                <td>{pct(r.prob_up, 0)}</td>
                <td>{rangeLabel(r.ret5d_low, r.ret5d_high)}</td>
                <td>{pct(r.vol20d, 1)}</td>
              </tr>
            ))}
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
