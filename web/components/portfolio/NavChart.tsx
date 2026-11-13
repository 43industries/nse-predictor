"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { NavPoint } from "../../lib/api";
import { kes } from "../../lib/format";

export function NavChart({ points }: { points: NavPoint[] }) {
  if (!points.length) {
    return <p className="muted">No NAV history yet — confirm a paper fill to start the series.</p>;
  }
  return (
    <div className="panel chart-wrap-md">
      <ResponsiveContainer>
        <LineChart data={points} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d5dde6" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={24} />
          <YAxis
            tick={{ fontSize: 11 }}
            width={64}
            tickFormatter={(v) => kes(Number(v), 0)}
          />
          <Tooltip
            formatter={(value: number, name: string) => [kes(value, 0), name]}
            labelFormatter={(l) => String(l)}
          />
          <Line type="monotone" dataKey="nav_kes" name="NAV" stroke="#0f2744" strokeWidth={2} dot={false} />
          <Line
            type="monotone"
            dataKey="equity_kes"
            name="Equity"
            stroke="#2f7d4a"
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
