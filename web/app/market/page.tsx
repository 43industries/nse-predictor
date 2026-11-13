"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Disclaimer } from "../../components/Disclaimer";
import { ForecastBoard } from "../../components/ForecastBoard";
import { MoversStrip } from "../../components/MoversStrip";
import { SignalFilters } from "../../components/SignalFilters";
import { StatusBanners } from "../../components/StatusBanners";
import { ApiError, api, type Forecast, type MarketMovers } from "../../lib/api";
import { pct } from "../../lib/format";
import { ticketFromSuggestion } from "../../lib/paperTicket";
import {
  PREF_KEYS,
  loadNumber,
  loadStringList,
  saveNumber,
  saveStringList,
  toggleInList,
} from "../../lib/prefs";

export default function MarketPage() {
  const [rows, setRows] = useState<Forecast[]>([]);
  const [movers, setMovers] = useState<MarketMovers | null>(null);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [disclaimer, setDisclaimer] = useState("");
  const [search, setSearch] = useState("");
  const [minProb, setMinProb] = useState(0);
  const [callFilter, setCallFilter] = useState<"all" | "up" | "down" | "flat">("all");
  const [favOnly, setFavOnly] = useState(false);
  const [favourites, setFavourites] = useState<string[]>([]);

  useEffect(() => {
    setFavourites(loadStringList(PREF_KEYS.marketFavourites));
    setMinProb(loadNumber(PREF_KEYS.marketMinProb, 0));
  }, []);

  const load = useCallback(async () => {
    try {
      setError("");
      const [board, mov] = await Promise.all([api.marketBoard(), api.marketMovers(5)]);
      setRows(board.rows || []);
      setMovers(mov);
      setDisclaimer(board.disclaimer || mov.disclaimer || "");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load market board");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function persistFavs(next: string[]) {
    setFavourites(next);
    saveStringList(PREF_KEYS.marketFavourites, next);
  }

  function setMinProbPersist(v: number) {
    setMinProb(v);
    saveNumber(PREF_KEYS.marketMinProb, v);
  }

  const filtered = useMemo(() => {
    const q = search.trim().toUpperCase();
    const favSet = new Set(favourites);
    return rows.filter((r) => {
      const code = r.stock_code.toUpperCase();
      if (q && !code.includes(q)) return false;
      if (favOnly && !favSet.has(code)) return false;
      if ((r.prob_up ?? 0) < minProb) return false;
      const call = String(r.direction_5d || r.pred_label || "").toLowerCase();
      if (callFilter !== "all" && call !== callFilter) return false;
      return true;
    });
  }, [rows, search, favOnly, favourites, minProb, callFilter]);

  async function onTicket(row: Forecast) {
    const code = row.stock_code.toUpperCase();
    setBusy(`${code}:ticket`);
    setMsg("");
    setError("");
    try {
      const res = await ticketFromSuggestion(row, "market_board");
      setMsg(
        `Pending paper ticket ${res.ticket.id}: ${row.suggested_side} ${code} × ${row.suggested_shares}`
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Ticket failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <main>
      <p className="brand-tag">Market board</p>
      <h1 className="hero-brand">
        All <span>forecasts</span>
      </h1>
      <p className="lede">
        Full investable board with P(up) ranking, session movers, and paper ticket hints.{" "}
        <strong>Ticket</strong> creates a pending paper order — confirm on{" "}
        <Link href="/portfolio" className="ticker-link">
          Portfolio
        </Link>
        .
      </p>

      <StatusBanners error={error} msg={msg} portfolioLink />

      <MoversStrip data={movers} />

      <SignalFilters
        minProb={minProb}
        showHit={false}
        onMinProb={setMinProbPersist}
        onRefresh={() => load()}
      >
        <label>
          Search
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="SCOM"
          />
        </label>
        <label>
          Call
          <select
            value={callFilter}
            onChange={(e) => setCallFilter(e.target.value as typeof callFilter)}
          >
            <option value="all">All</option>
            <option value="up">Up</option>
            <option value="down">Down</option>
            <option value="flat">Flat</option>
          </select>
        </label>
        <label className="check-label">
          <input
            type="checkbox"
            checked={favOnly}
            onChange={(e) => setFavOnly(e.target.checked)}
          />
          Favourites only ({favourites.length})
        </label>
      </SignalFilters>

      <p className="muted">
        Showing {filtered.length} of {rows.length} · Min P(up) {pct(minProb, 0)}
      </p>

      {!error && (
        <ForecastBoard
          rows={filtered}
          favourites={favourites}
          onToggleFavourite={(code) => persistFavs(toggleInList(favourites, code))}
          onTicket={onTicket}
          busy={busy}
        />
      )}
      <Disclaimer text={disclaimer} />
    </main>
  );
}
