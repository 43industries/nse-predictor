"use client";

import { useCallback, useEffect, useState } from "react";
import { Disclaimer } from "../components/Disclaimer";
import { KpiStrip } from "../components/KpiStrip";
import { SetupTable } from "../components/SetupTable";
import { SkipList } from "../components/SkipList";
import { StockLink } from "../components/StockLink";
import {
  ApiError,
  api,
  type DeskBrief,
  type JournalEntry,
  type SetupRow,
} from "../lib/api";
import { pct } from "../lib/format";

export default function DeskBriefPage() {
  const [brief, setBrief] = useState<DeskBrief | null>(null);
  const [journal, setJournal] = useState<JournalEntry[]>([]);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState("");

  const load = useCallback(async () => {
    try {
      setError("");
      const [b, j] = await Promise.all([api.deskBrief(), api.journal()]);
      setBrief(b);
      setJournal(j.entries || []);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load desk brief");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

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

  return (
    <main>
      <p className="brand-tag">Nairobi Securities Exchange</p>
      <h1 className="hero-brand">
        NSE <span>Predictor</span>
      </h1>
      <p className="lede">
        Today&apos;s desk brief — ranked setups, honest skips, and a paper journal. Research companion
        only; you decide.
      </p>

      {error && <div className="error-banner">{error}</div>}
      {msg && <p className="muted">{msg}</p>}

      {brief && (
        <>
          <KpiStrip
            items={[
              { label: "As of", value: brief.as_of || "—" },
              { label: "Setups", value: String(brief.n_setups) },
              { label: "Skips", value: String(brief.n_skips) },
              { label: "Model thr", value: Number(brief.threshold).toFixed(2) },
            ]}
          />

          <p className="section-label">Today&apos;s setups</p>
          <input
            className="note-input"
            placeholder="Optional journal note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <SetupTable rows={brief.setups} onLog={onLog} busy={busy} />

          <p className="section-label">Skips</p>
          <SkipList skips={brief.skips} />

          <p className="section-label">Paper journal</p>
          <div className="panel" style={{ overflowX: "auto" }}>
            {journal.length === 0 ? (
              <p className="muted">No paper entries yet. Use Log long / Log skip on a setup.</p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Logged</th>
                    <th>As of</th>
                    <th>Stock</th>
                    <th>Stance</th>
                    <th>P(up)</th>
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
