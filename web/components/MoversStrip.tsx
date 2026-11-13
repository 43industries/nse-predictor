"use client";

import type { MarketMovers, MoverRow } from "../lib/api";
import { pct } from "../lib/format";
import { CompanyLogo } from "./CompanyLogo";
import { StockLink } from "./StockLink";

function Board({ title, rows }: { title: string; rows: MoverRow[] }) {
  return (
    <div className="panel mover-box">
      <p className="section-label" style={{ marginTop: 0 }}>
        {title}
      </p>
      {!rows.length ? (
        <p className="muted small">No rows</p>
      ) : (
        <ul className="mover-list">
          {rows.map((r) => {
            const cls = r.ret_1d > 0 ? "up" : r.ret_1d < 0 ? "down" : "muted";
            return (
              <li key={`${title}-${r.stock_code}`}>
                <StockLink code={r.stock_code} className="ticker-cell">
                  <CompanyLogo code={r.stock_code} />
                  <strong>{r.stock_code}</strong>
                </StockLink>
                <span className="muted">
                  {r.price != null ? Number(r.price).toFixed(2) : "—"}
                </span>
                <span className={cls}>{pct(r.ret_1d, 1)}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export function MoversStrip({ data }: { data: MarketMovers | null }) {
  if (!data) return null;
  return (
    <div>
      <p className="section-label">Session movers</p>
      <p className="muted small">
        As of {data.as_of || "—"}. {data.note || "Price moves — not model signals."}
      </p>
      <div className="movers-grid">
        <Board title="Gainers" rows={data.gainers || []} />
        <Board title="Losers" rows={data.losers || []} />
        <Board title="Movers" rows={data.movers || []} />
      </div>
    </div>
  );
}
