"use client";

import Link from "next/link";
import type { DeskBriefSummary, DeskWatchlistItem, SetupRow } from "../lib/api";
import { callLabel, pct, rangeLabel, stockPath } from "../lib/format";
import { CompanyLogo } from "./CompanyLogo";
import { StockLink } from "./StockLink";

type Props = {
  rows: SetupRow[];
  summary?: DeskBriefSummary | null;
  watchlist?: DeskWatchlistItem[];
  onLog?: (row: SetupRow, stance: "long" | "skip") => void;
  onTicket?: (row: SetupRow) => void;
  busy?: string | null;
};

export function SetupTable({
  rows,
  summary,
  watchlist = [],
  onLog,
  onTicket,
  busy,
}: Props) {
  if (!rows.length) {
    return (
      <div className="empty-state panel">
        <p className="empty-state-title">No buy setups today</p>
        {summary ? (
          <ul className="empty-state-list">
            <li>
              {summary.n_up} up · {summary.n_flat} flat · {summary.n_down} down calls
            </li>
            <li>Minimum confidence to act: {pct(summary.entry_threshold, 0)}</li>
            {summary.n_at_chance > 0 ? (
              <li>{summary.n_at_chance} names marked unreliable (at-chance)</li>
            ) : null}
          </ul>
        ) : (
          <p className="muted">Nothing passed the confidence filters.</p>
        )}

        {watchlist.length > 0 ? (
          <div className="watchlist">
            <p className="section-label" style={{ marginTop: "0.75rem" }}>
              Watchlist (highest chance up)
            </p>
            <ul className="watchlist-list">
              {watchlist.map((w) => (
                <li key={w.stock_code}>
                  <StockLink code={w.stock_code} />{" "}
                  <span className="muted">
                    {pct(w.prob_up, 0)} · {callLabel(w.pred_label)}
                    {w.skip_label ? ` · ${w.skip_label}` : w.in_setups ? " · setup" : ""}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <p className="empty-state-cta">
          <Link href="/market" className="btn">
            Open market board
          </Link>
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <table className="data-table">
        <thead>
          <tr>
            <th>Stock</th>
            <th>Price</th>
            <th>Call</th>
            <th>Chance up</th>
            <th>Range</th>
            <th>Recent hit</th>
            <th>Book</th>
            <th>Paper size</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const canTicket =
              !!onTicket && !!r.suggested_side && (r.suggested_shares || 0) > 0;
            return (
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
                  <span className="pill">{r.book_status || "flat"}</span>
                  {r.held_shares ? ` ×${r.held_shares}` : ""}
                </td>
                <td>
                  {canTicket
                    ? `${String(r.suggested_side).toUpperCase()} ${r.suggested_shares}`
                    : r.suggest_note || "—"}
                </td>
                <td>
                  <div className="btn-row">
                    {canTicket && (
                      <button
                        className="btn btn-primary"
                        disabled={busy === `${r.stock_code}:ticket`}
                        onClick={() => onTicket?.(r)}
                        title="Creates a pending paper ticket — not a live order"
                      >
                        Ticket
                      </button>
                    )}
                    <Link className="btn" href={stockPath(r.stock_code)}>
                      Open
                    </Link>
                    {onLog && (
                      <details className="row-journal">
                        <summary className="muted">Journal</summary>
                        <div className="btn-row" style={{ marginTop: "0.35rem" }}>
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
                        </div>
                      </details>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
