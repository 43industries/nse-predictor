"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Forecast } from "../lib/api";
import { callLabel, pct, rangeLabel } from "../lib/format";
import { KpiStrip } from "./KpiStrip";

type HistPoint = { date: string; price: number; volume?: number };
type Driver = { feature?: string; mean_abs_shap?: number };

type Props = {
  forecast: Forecast;
  hitRate?: number | null;
  history: HistPoint[];
  drivers: Driver[];
};

function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function buildBand(history: HistPoint[], lo?: number | null, hi?: number | null) {
  if (!history.length || lo == null || hi == null) return null;
  const last = history[history.length - 1];
  if (last?.price == null) return null;
  return {
    x1: last.date,
    x2: addDays(last.date, 7),
    y1: last.price * (1 + Number(lo)),
    y2: last.price * (1 + Number(hi)),
    mid: last.price * (1 + (Number(lo) + Number(hi)) / 2),
  };
}

export function StockBrief({ forecast, hitRate, history, drivers }: Props) {
  const band = useMemo(
    () => buildBand(history, forecast.ret5d_low, forecast.ret5d_high),
    [history, forecast.ret5d_low, forecast.ret5d_high]
  );

  const chartData = useMemo(() => {
    if (!history.length) return [];
    const rows = history.map((p) => ({
      date: p.date,
      price: p.price,
      volume: p.volume ?? 0,
    }));
    if (band) {
      rows.push({ date: band.x2, price: Number.NaN, volume: 0 });
    }
    return rows;
  }, [history, band]);

  const shapRows = useMemo(() => {
    const top = drivers.slice(0, 10);
    const total = top.reduce((s, d) => s + Number(d.mean_abs_shap || 0), 0) || 1;
    return [...top]
      .map((d) => ({
        feature: String(d.feature || "—"),
        share: (Number(d.mean_abs_shap || 0) / total) * 100,
        abs: Number(d.mean_abs_shap || 0),
      }))
      .reverse();
  }, [drivers]);

  return (
    <div>
      <KpiStrip
        items={[
          { label: "As of", value: forecast.as_of_date || "—" },
          {
            label: "Price",
            value: forecast.price != null ? Number(forecast.price).toFixed(2) : "—",
          },
          { label: "P(up)", value: pct(forecast.prob_up, 0) },
          {
            label: "Call",
            value: callLabel(forecast.direction_5d, forecast.pred_label),
          },
        ]}
      />

      <div className="panel" style={{ marginTop: "1rem" }}>
        <p className="section-label" style={{ marginTop: 0 }}>
          Price · volume · 5d range
        </p>
        {chartData.length === 0 ? (
          <p className="muted">No price history for this ticker.</p>
        ) : (
          <>
            <div className="chart-wrap chart-wrap-lg">
              <ResponsiveContainer>
                <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#e8eaed" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#5c6b7a" }} minTickGap={28} />
                  <YAxis
                    yAxisId="price"
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 11, fill: "#5c6b7a" }}
                    width={48}
                  />
                  <YAxis
                    yAxisId="vol"
                    orientation="right"
                    domain={[0, "auto"]}
                    tick={{ fontSize: 10, fill: "#5c6b7a" }}
                    width={40}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => {
                      if (name === "volume") return [Math.round(value).toLocaleString(), "Volume"];
                      if (name === "price") return [Number(value).toFixed(2), "Price"];
                      return [value, name];
                    }}
                  />
                  {band && (
                    <ReferenceArea
                      yAxisId="price"
                      x1={band.x1}
                      x2={band.x2}
                      y1={band.y1}
                      y2={band.y2}
                      fill="rgba(47, 125, 74, 0.16)"
                      strokeOpacity={0}
                    />
                  )}
                  {band && (
                    <ReferenceLine
                      yAxisId="price"
                      y={band.mid}
                      stroke="#2f7d4a"
                      strokeDasharray="4 4"
                      strokeWidth={1}
                    />
                  )}
                  <Bar
                    yAxisId="vol"
                    dataKey="volume"
                    fill="#c5d0db"
                    opacity={0.85}
                    barSize={4}
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="price"
                    type="monotone"
                    dataKey="price"
                    stroke="#0f2744"
                    dot={false}
                    strokeWidth={2}
                    connectNulls={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <p className="muted">
              Green band = model 5d price range from last close · Range{" "}
              {rangeLabel(forecast.ret5d_low, forecast.ret5d_high)} · Vol {pct(forecast.vol20d, 1)} ·
              Hit rate {pct(hitRate, 0)}
            </p>
          </>
        )}
      </div>

      <div className="panel" style={{ marginTop: "1rem" }}>
        <p className="section-label" style={{ marginTop: 0 }}>
          Global drivers (SHAP)
        </p>
        {shapRows.length === 0 ? (
          <p className="muted">No SHAP drivers yet.</p>
        ) : (
          <div className="chart-wrap chart-wrap-md">
            <ResponsiveContainer>
              <BarChart
                data={shapRows}
                layout="vertical"
                margin={{ left: 8, right: 16, top: 4, bottom: 4 }}
              >
                <CartesianGrid stroke="#e8eaed" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11, fill: "#5c6b7a" }}
                  unit="%"
                  domain={[0, "auto"]}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={110}
                  tick={{ fontSize: 11, fill: "#5c6b7a" }}
                />
                <Tooltip
                  formatter={(v: number, _n, item) => [
                    `${Number(v).toFixed(1)}% (${Number(item?.payload?.abs || 0).toFixed(4)})`,
                    "Share of top drivers",
                  ]}
                />
                <Bar dataKey="share" fill="#2f7d4a" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
