"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Disclaimer } from "../../components/Disclaimer";
import { KpiStrip } from "../../components/KpiStrip";
import { ManualTicketForm } from "../../components/portfolio/ManualTicketForm";
import { NavChart } from "../../components/portfolio/NavChart";
import { RiskPanel } from "../../components/portfolio/RiskPanel";
import { SeedPositionForm } from "../../components/portfolio/SeedPositionForm";
import { SignalFilters } from "../../components/SignalFilters";
import { StatusBanners } from "../../components/StatusBanners";
import { StockLink } from "../../components/StockLink";
import {
  ApiError,
  api,
  type NavPoint,
  type PortfolioBook,
  type PortfolioFill,
  type PortfolioSuggestion,
  type PortfolioSuggestions,
  type PortfolioTicket,
} from "../../lib/api";
import { downloadCsv } from "../../lib/csv";
import { kes, pct } from "../../lib/format";
import { PREF_KEYS, loadNumber, saveNumber } from "../../lib/prefs";

const ACTIONABLE = new Set(["open", "add", "trim", "exit"]);

export default function PortfolioPage() {
  const [book, setBook] = useState<PortfolioBook | null>(null);
  const [suggestions, setSuggestions] = useState<PortfolioSuggestions | null>(null);
  const [tickets, setTickets] = useState<PortfolioTicket[]>([]);
  const [fills, setFills] = useState<PortfolioFill[]>([]);
  const [navPoints, setNavPoints] = useState<NavPoint[]>([]);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [resetCash, setResetCash] = useState(1_000_000);
  const [showFills, setShowFills] = useState(false);
  const [minProb, setMinProb] = useState(0.55);
  const [minHit, setMinHit] = useState(0.52);

  useEffect(() => {
    setMinProb(loadNumber(PREF_KEYS.minProbUp, 0.55));
    setMinHit(loadNumber(PREF_KEYS.minHitRate, 0.52));
  }, []);

  const load = useCallback(async () => {
    try {
      setError("");
      const [b, s, t, f, nav] = await Promise.all([
        api.portfolio(),
        api.portfolioSuggestions({ min_prob_up: minProb, min_hit_rate: minHit }),
        api.portfolioTickets("pending"),
        api.portfolioFills(40),
        api.portfolioNavHistory(90),
      ]);
      setBook(b);
      setSuggestions(s);
      setTickets(t.tickets || []);
      setFills(f.fills || []);
      setNavPoints(nav.points || []);
      if (b.cash_kes != null && b.n_names === 0) {
        setResetCash(Math.round(Number(b.cash_kes)) || 1_000_000);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load portfolio");
    }
  }, [minProb, minHit]);

  useEffect(() => {
    load();
  }, [load]);

  const actionable = useMemo(
    () =>
      (suggestions?.suggestions || []).filter(
        (s) => ACTIONABLE.has(s.action) && (s.shares || 0) > 0 && s.side
      ),
    [suggestions]
  );

  async function run(key: string, fn: () => Promise<void>) {
    try {
      setBusy(key);
      setMsg("");
      setError("");
      await fn();
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Action failed");
    } finally {
      setBusy(null);
    }
  }

  function persistFilters(nextProb: number, nextHit: number) {
    setMinProb(nextProb);
    setMinHit(nextHit);
    saveNumber(PREF_KEYS.minProbUp, nextProb);
    saveNumber(PREF_KEYS.minHitRate, nextHit);
  }

  function ticketFromSuggestion(s: PortfolioSuggestion) {
    if (!s.side || !s.shares) return;
    return run(`sug-${s.stock_code}-${s.action}`, async () => {
      await api.portfolioCreateTicket({
        stock_code: s.stock_code,
        side: s.side as "buy" | "sell",
        shares: s.shares,
        reason: s.reason || "",
        suggestion_action: s.action,
      });
      setMsg(`Pending ${s.side} ticket: ${s.stock_code} × ${s.shares}`);
    });
  }

  return (
    <main>
      <p className="brand-tag">Paper portfolio</p>
      <h1 className="hero-brand">
        Portfolio <span>desk</span>
      </h1>
      <p className="lede">
        Research paper book only — suggestions are not investment advice and do not route live
        orders. Confirm tickets yourself.
      </p>

      <StatusBanners error={error} msg={msg} />

      {book && (
        <KpiStrip
          items={[
            { label: "NAV (KES)", value: kes(book.nav_kes) },
            { label: "Cash", value: kes(book.cash_kes) },
            { label: "Equity", value: kes(book.equity_kes) },
            { label: "Unrealized P&L", value: kes(book.unrealized_pnl_kes) },
            { label: "Names", value: String(book.n_names ?? 0) },
          ]}
        />
      )}

      <RiskPanel risk={book?.risk} />

      <SignalFilters
        minProb={minProb}
        minHit={minHit}
        onMinProb={(v) => persistFilters(v, minHit)}
        onMinHit={(v) => persistFilters(minProb, v)}
        onRefresh={() => load()}
      />

      <details className="panel portfolio-reset">
        <summary>Reset paper book</summary>
        <div className="btn-row" style={{ alignItems: "center", marginTop: "0.75rem" }}>
          <label className="muted">
            Starting cash{" "}
            <input
              type="number"
              min={10000}
              step={50000}
              value={resetCash}
              onChange={(e) => setResetCash(Number(e.target.value))}
            />
          </label>
          <button
            className="btn"
            disabled={busy === "reset"}
            onClick={() =>
              run("reset", async () => {
                await api.portfolioReset(resetCash);
                setMsg("Paper book reset.");
              })
            }
          >
            {busy === "reset" ? "Resetting…" : "Reset"}
          </button>
        </div>
      </details>

      <SeedPositionForm
        busy={busy === "seed"}
        onSubmit={(body) =>
          run("seed", async () => {
            await api.portfolioSeedPosition(body);
            setMsg(`Seeded ${body.stock_code} × ${body.shares}`);
          })
        }
      />

      <ManualTicketForm
        lotSize={book?.lot_size || 100}
        busy={busy === "manual"}
        onSubmit={(body) =>
          run("manual", async () => {
            await api.portfolioCreateTicket(body);
            setMsg(`Pending ${body.side} ${body.stock_code} × ${body.shares}`);
          })
        }
      />

      <p className="section-label">Positions</p>
      {!book?.positions?.length ? (
        <p className="muted">No positions yet. Seed holdings or confirm a buy ticket.</p>
      ) : (
        <div className="panel">
          <table className="data-table">
            <thead>
              <tr>
                <th>Code</th>
                <th>Shares</th>
                <th>Avg cost</th>
                <th>Last</th>
                <th>Mkt value</th>
                <th>P&L</th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody>
              {book.positions.map((p) => (
                <tr key={p.stock_code}>
                  <td>
                    <StockLink code={p.stock_code} />
                  </td>
                  <td>{p.shares}</td>
                  <td>{kes(p.avg_cost, 2)}</td>
                  <td>{kes(p.last_price, 2)}</td>
                  <td>{kes(p.market_value)}</td>
                  <td className={(p.unrealized_pnl ?? 0) >= 0 ? "up" : "down"}>
                    {kes(p.unrealized_pnl)}
                  </td>
                  <td>{pct(p.weight, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="section-head">
        <p className="section-label">Suggestions</p>
        <button
          className="btn btn-accent"
          type="button"
          disabled={!actionable.length || !!busy}
          onClick={() =>
            run("batch", async () => {
              const payload = actionable.map((s) => ({
                stock_code: s.stock_code,
                side: s.side as "buy" | "sell",
                shares: s.shares,
                reason: s.reason || "",
                suggestion_action: s.action,
              }));
              const res = await api.portfolioCreateTicketsBatch(payload);
              setMsg(
                `Batch: ${res.n_created} created` +
                  (res.n_errors ? `, ${res.n_errors} failed` : "")
              );
            })
          }
        >
          {busy === "batch" ? "…" : "Ticket all actionable"}
        </button>
      </div>
      <p className="muted">
        As of {suggestions?.as_of || "—"} · {suggestions?.n_suggestions ?? 0} actionable · lot=
        {book?.lot_size ?? "—"} · filters P(up)≥{pct(minProb, 0)} hit≥{pct(minHit, 0)}
      </p>
      {!actionable.length ? (
        <p className="muted">No actionable suggestions vs current book / filters.</p>
      ) : (
        <div className="panel suggestion-list">
          {actionable.map((s) => {
            const key = `${s.stock_code}-${s.action}-${s.shares}`;
            return (
              <div className="suggestion-row" key={key}>
                <div>
                  <strong>
                    <StockLink code={s.stock_code} />
                  </strong>{" "}
                  <span className="pill">{s.action}</span>
                </div>
                <div className="muted">
                  P(up) {pct(s.prob_up, 0)}
                  {s.hit_rate != null ? ` · hit ${pct(s.hit_rate, 0)}` : ""} · {s.shares} sh ·{" "}
                  {s.side || "—"}
                  {s.est_notional_kes != null ? ` · ~${kes(s.est_notional_kes)}` : ""}
                </div>
                <div className="muted small">{s.reason}</div>
                <button
                  className="btn btn-accent"
                  disabled={!!busy}
                  onClick={() => ticketFromSuggestion(s)}
                >
                  {busy === `sug-${s.stock_code}-${s.action}` ? "…" : "Ticket"}
                </button>
              </div>
            );
          })}
        </div>
      )}

      <div className="section-head">
        <p className="section-label">Pending tickets</p>
        <button
          className="btn"
          type="button"
          disabled={!tickets.length}
          onClick={() =>
            downloadCsv(
              `portfolio_tickets_${new Date().toISOString().slice(0, 10)}.csv`,
              tickets as unknown as Record<string, unknown>[]
            )
          }
        >
          Export CSV
        </button>
      </div>
      {!tickets.length ? (
        <p className="muted">No pending tickets.</p>
      ) : (
        <div className="panel suggestion-list">
          {tickets.map((t) => (
            <div className="suggestion-row" key={t.id}>
              <div>
                <code>{t.id}</code> · <strong>{String(t.side).toUpperCase()}</strong>{" "}
                <StockLink code={t.stock_code} /> × {t.shares}
              </div>
              <div className="muted small">{t.reason || t.suggestion_action || ""}</div>
              <div className="btn-row">
                <button
                  className="btn btn-primary"
                  disabled={!!busy}
                  onClick={() =>
                    run(`confirm-${t.id}`, async () => {
                      await api.portfolioConfirmTicket(t.id);
                      setMsg(`Filled ${t.stock_code} (paper).`);
                    })
                  }
                >
                  {busy === `confirm-${t.id}` ? "…" : "Confirm fill"}
                </button>
                <button
                  className="btn"
                  disabled={!!busy}
                  onClick={() =>
                    run(`reject-${t.id}`, async () => {
                      await api.portfolioRejectTicket(t.id);
                      setMsg(`Rejected ${t.id}`);
                    })
                  }
                >
                  Reject
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="section-label">NAV history</p>
      <NavChart points={navPoints} />

      <div className="section-head">
        <p className="section-label">Fill ledger</p>
        <div className="btn-row">
          <button className="btn" type="button" onClick={() => setShowFills((v) => !v)}>
            {showFills ? "Hide fills" : "Show fills"}
          </button>
          <button
            className="btn"
            type="button"
            disabled={!fills.length}
            onClick={() =>
              downloadCsv(
                `portfolio_fills_${new Date().toISOString().slice(0, 10)}.csv`,
                fills as unknown as Record<string, unknown>[]
              )
            }
          >
            Export fills CSV
          </button>
        </div>
      </div>
      {showFills && (
        <div className="panel" style={{ marginTop: "0.75rem" }}>
          {!fills.length ? (
            <p className="muted">No fills yet.</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Side</th>
                  <th>Code</th>
                  <th>Shares</th>
                  <th>Price</th>
                  <th>Fees</th>
                </tr>
              </thead>
              <tbody>
                {fills.map((f, i) => (
                  <tr key={`${f.ticket_id}-${i}`}>
                    <td>{(f.ts || "").slice(0, 19)}</td>
                    <td>{f.side}</td>
                    <td>
                      <StockLink code={f.stock_code} />
                    </td>
                    <td>{f.shares}</td>
                    <td>{kes(f.price, 2)}</td>
                    <td>{kes(f.fees_kes, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <Disclaimer text={book?.disclaimer || suggestions?.disclaimer || ""} />
    </main>
  );
}
