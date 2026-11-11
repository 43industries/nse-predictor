"use client";

import { useEffect, useState } from "react";
import { Disclaimer } from "../../components/Disclaimer";
import { ForecastBoard } from "../../components/ForecastBoard";
import { ApiError, api, type Forecast } from "../../lib/api";

export default function MarketPage() {
  const [rows, setRows] = useState<Forecast[]>([]);
  const [error, setError] = useState("");
  const [disclaimer, setDisclaimer] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const [fc, d] = await Promise.all([api.forecasts(), api.disclaimer()]);
        setRows(fc);
        setDisclaimer(d.disclaimer || "");
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Failed to load forecasts");
      }
    }
    load();
  }, []);

  return (
    <main>
      <p className="brand-tag">Market board</p>
      <h1 className="hero-brand">
        All <span>forecasts</span>
      </h1>
      <p className="lede">Full investable board with P(up) ranking. Click a ticker for detail.</p>
      {error && <div className="error-banner">{error}</div>}
      {!error && rows.length > 0 && <ForecastBoard rows={rows} />}
      <Disclaimer text={disclaimer} />
    </main>
  );
}
