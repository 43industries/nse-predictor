"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Disclaimer } from "../components/Disclaimer";
import { KpiStrip } from "../components/KpiStrip";
import { SetupTable } from "../components/SetupTable";
import {
  FILTER_PRESETS,
  SignalFilters,
  type FilterPreset,
} from "../components/SignalFilters";
import { SkipList } from "../components/SkipList";
import { StatusBanners } from "../components/StatusBanners";
import { StockLink } from "../components/StockLink";
import {
  ApiError,
  api,
  type DeskBrief,
  type JournalEntry,
  type SetupRow,
} from "../lib/api";
import { pct } from "../lib/format";
import { ticketFromSuggestion } from "../lib/paperTicket";
import { PREF_KEYS, loadNumber, saveNumber } from "../lib/prefs";

export default function DeskBriefPage() {
  const [brief, setBrief] = useState<DeskBrief | null>(null);
  const [journal, setJournal] = useState<JournalEntry[]>([]);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const [minProb, setMinProb] = useState(0.55);
  const [minHit, setMinHit] = useState(0.52);

  useEffect(() => {
    setMinProb(loadNumber(PREF_KEYS.minProbUp, 0.55));
    setMinHit(loadNumber(PREF_KEYS.minHitRate, 0.52));
  }, []);

  const load = useCallback(async () => {
    try {
      setError("");
      const [b, j] = await Promise.all([
        api.deskBrief({ top_n: 10, min_prob_up: minProb, min_hit_rate: minHit }),
        api.journal(),
      ]);
      setBrief(b);
      setJournal(j.entries || []);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load desk brief");
    }
  }, [minProb, minHit]);

  useEffect(() => {
    load();
  }, [load]);

  function persistFilters(nextProb: number, nextHit: number) {
    setMinProb(nextProb);
    setMinHit(nextHit);
    saveNumber(PREF_KEYS.minProbUp, nextProb);
    saveNumber(PREF_KEYS.minHitRate, nextHit);
  }

  function applyPreset(preset: Exclude<FilterPreset, "custom">) {
    const p = FILTER_PRESETS[preset];
    persistFilters(p.minProb, p.minHit);
  }

  async function onLog(row: SetupRow, stance: "long" | "skip") {
    setBusy(`${row.stock_code}:${stance}`);
    setMsg("");
    try {
      await api.logJournal({
        stock_code: row.stock_code,
        stance,
        note,
        as_of_date: brief?.as_of,
        prob_up: row.prob_up,
        pred_label: row.pred_label,
        price: row.price,
      });
      setMsg(`Logged ${stance} ${row.stock_code}`);
      const j = await api.journal();
      setJournal(j.entries || []);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Journal write failed");
    } finally {
      setBusy(null);
    }
  }

  async function onTicket(row: SetupRow) {
    setBusy(`${row.stock_code}:ticket`);
    setMsg("");
    setError("");
    try {
      const res = await ticketFromSuggestion(row, "desk_brief");
      setMsg(
        `Pending paper ticket ${res.ticket.id}: ${row.suggested_side} ${row.stock_code} × ${row.suggested_shares}`
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Ticket failed");
    } finally {
      setBusy(null);
    }
  }

  const summary = brief?.summary;
  const entryThr = Number(summary?.entry_threshold ?? brief?.filters?.entry_threshold ?? brief?.threshold ?? 0);

  return (
    <main>
      <p className="brand-tag">Nairobi Securities Exchange</p>
      <h1 className="hero-brand">
        NSE <span>Predictor</span>
      </h1>
      <p className="lede">
        Today&apos;s desk brief — ranked setups, book-aware sizing hints, and a paper journal.
        Research companion only. <strong>Ticket</strong> creates a pending paper order (not live) —
        confirm on{" "}
        <Link href="/portfolio" className="ticker-link">
          Portfolio
        </Link>
        .
      </p>

      <StatusBanners error={error} msg={msg} portfolioLink={msg.includes("Pending paper ticket")} />

      {brief && summary ? (
        <p className="desk-status">
          <strong>As of {summary.as_of || brief.as_of || "—"}.</strong> {summary.headline}{" "}
          <span className="muted">
            Minimum confidence to act: {pct(entryThr, 0)}.
          </span>
        </p>
      ) : null}

      {summary?.model_caution ? (
        <div className="caution-banner">
          Model is cautious today — many names are marked unreliable. Prefer{" "}
          <Link href="/market" className="ticker-link">
            Market
          </Link>{" "}
          or{" "}
          <Link href="/portfolio" className="ticker-link">
            Portfolio
          </Link>{" "}
          over forcing tickets.
        </div>
      ) : null}

      <SignalFilters
        minProb={minProb}
        minHit={minHit}
        showPresets
        onMinProb={(v) => persistFilters(v, minHit)}
        onMinHit={(v) => persistFilters(minProb, v)}
        onPreset={applyPreset}
        onRefresh={() => load()}
      />

      {brief && (
        <>
          <KpiStrip
            items={[
              { label: "As of", value: brief.as_of || "—" },
              { label: "Buy setups", value: String(brief.n_setups) },
              { label: "Skips", value: String(brief.n_skips) },
              {
                label: "Confidence bar",
                value: pct(entryThr, 0),
              },
            ]}
          />

          <p className="section-label">Today&apos;s setups</p>
          <input
            className="note-input"
            placeholder="Optional journal note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <SetupTable
            rows={brief.setups}
            summary={summary}
            watchlist={brief.watchlist}
            onLog={onLog}
            onTicket={onTicket}
            busy={busy}
          />

          <p className="section-label">Skips</p>
          <SkipList skips={brief.skips} />

          <p className="section-label">Paper journal</p>
          <div className="panel" style={{ overflowX: "auto" }}>
            {journal.length === 0 ? (
              <p className="muted">
                No paper entries yet. Open Journal on a setup to log long or skip.
              </p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Logged</th>
                    <th>As of</th>
                    <th>Stock</th>
                    <th>Stance</th>
                    <th>Chance up</th>
                    <th>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {journal.slice(0, 30).map((e, i) => (
                    <tr key={`${e.logged_at}-${e.stock_code}-${i}`}>
                      <td>{e.logged_at ? String(e.logged_at).slice(0, 19) : "—"}</td>
                      <td>{e.as_of_date || "—"}</td>
                      <td>
                        <StockLink code={e.stock_code} />
                      </td>
                      <td>{e.stance}</td>
                      <td>{pct(e.prob_up, 0)}</td>
                      <td>{e.note || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <Disclaimer text={brief.disclaimer} />
        </>
      )}
    </main>
  );
}
