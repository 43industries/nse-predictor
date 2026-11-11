"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Forecast } from "../lib/api";

function tone(f: Forecast): { cls: string; arrow: string; move: string } {
  const label = String(f.direction_5d || f.pred_label || "").toLowerCase();
  const prob = Number(f.prob_up ?? 0.5);
  if (label === "up") return { cls: "ticker-up", arrow: "▲", move: `+${(prob * 100).toFixed(0)}%` };
  if (label === "down") return { cls: "ticker-down", arrow: "▼", move: `${(prob * 100).toFixed(0)}%` };
  return { cls: "ticker-flat", arrow: "•", move: `${(prob * 100).toFixed(0)}%` };
}

export function TickerBar() {
  const [rows, setRows] = useState<Forecast[]>([]);

  useEffect(() => {
    let alive = true;
    api
      .forecasts()
      .then((data) => {
        if (!alive) return;
        const ordered = [...(data || [])].sort(
          (a, b) => Math.abs(Number(b.prob_up ?? 0.5) - 0.5) - Math.abs(Number(a.prob_up ?? 0.5) - 0.5)
        );
        setRows(ordered);
      })
      .catch(() => {
        if (alive) setRows([]);
      });
    return () => {
      alive = false;
    };
  }, []);

  if (!rows.length) return null;

  const items = [...rows, ...rows];

  return (
    <div className="ticker-bar" aria-label="Forecast tape">
      <div className="ticker-track">
        {items.map((f, i) => {
          const t = tone(f);
          const px = f.price != null && !Number.isNaN(Number(f.price)) ? Number(f.price).toFixed(2) : "—";
          return (
            <Link
              key={`${f.stock_code}-${i}`}
              href={`/stock/${f.stock_code}`}
              className="ticker-item"
            >
              <span className="ticker-code">{f.stock_code}</span>
              <span className="ticker-px">{px}</span>
              <span className={t.cls}>
                {t.arrow} {t.move}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
