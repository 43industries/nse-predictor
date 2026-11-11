"use client";

import { useEffect, useMemo, useState } from "react";
import { Disclaimer } from "../../components/Disclaimer";
import { MonitorPanel } from "../../components/health/MonitorPanel";
import { ValidationPanel } from "../../components/health/ValidationPanel";
import { ApiError, api } from "../../lib/api";
import { buildHealthView } from "../../lib/health";

export default function HealthPage() {
  const [evalJson, setEval] = useState<Record<string, unknown> | null>(null);
  const [collection, setCollection] = useState<Record<string, unknown> | null>(null);
  const [monitor, setMonitor] = useState<Record<string, unknown> | null>(null);
  const [monHist, setMonHist] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState("");
  const [disclaimer, setDisclaimer] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setError("");
        const [ev, col, mon, hist, d] = await Promise.all([
          api.evaluation(),
          api.collectionStatus().catch(() => ({})),
          api.monitor().catch(() => ({})),
          api.monitorHistory(14).catch(() => ({ rows: [] as Record<string, unknown>[] })),
          api.disclaimer(),
        ]);
        if (!alive) return;
        setEval(ev);
        setCollection(col);
        setMonitor(mon);
        setMonHist(hist.rows || []);
        setDisclaimer(d.disclaimer || "");
      } catch (e) {
        if (!alive) return;
        setError(e instanceof ApiError ? e.message : "Failed to load health");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const view = useMemo(
    () => buildHealthView(evalJson, monitor, collection),
    [evalJson, monitor, collection]
  );

  return (
    <main>
      <p className="brand-tag">Model health</p>
      <h1 className="hero-brand">
        Honesty <span>checks</span>
      </h1>
      <p className="lede">
        Full model status — monitor, validation scorecard, selection, and baselines. Research
        companion only.
      </p>
      {error && <div className="error-banner">{error}</div>}

      <MonitorPanel view={view} history={monHist} />
      {evalJson && <ValidationPanel view={view} />}
      <Disclaimer text={disclaimer} />
    </main>
  );
}
