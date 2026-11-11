"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { CompanyLogo } from "../../../components/CompanyLogo";
import { Disclaimer } from "../../../components/Disclaimer";
import { StockBrief } from "../../../components/StockBrief";
import { ApiError, api, type Forecast } from "../../../lib/api";

export default function StockPage() {
  const params = useParams();
  const code = String(params?.code || "").toUpperCase();
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [history, setHistory] = useState<{ date: string; price: number; volume?: number }[]>([]);
  const [hitRate, setHitRate] = useState<number | null>(null);
  const [drivers, setDrivers] = useState<{ feature?: string; mean_abs_shap?: number }[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!code) return;
    async function load() {
      try {
        setError("");
        const [f, h, a, d] = await Promise.all([
          api.forecast(code),
          api.history(code),
          api.accuracy(code).catch(() => ({ hit_rate: null })),
          api.drivers(),
        ]);
        setForecast(f);
        setHistory(h.points || []);
        setHitRate(a.hit_rate ?? null);
        setDrivers(d.drivers || []);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Failed to load stock");
      }
    }
    load();
  }, [code]);

  return (
    <main>
      <p className="brand-tag">Stock detail</p>
      <div className="hero-with-logo">
        {code ? <CompanyLogo code={code} size="lg" /> : null}
        <h1 className="hero-brand">{code || "—"}</h1>
      </div>
      <p className="lede">5d excess-vs-NASI signal, range, history, and model drivers.</p>
      {error && <div className="error-banner">{error}</div>}
      {forecast && (
        <>
          <StockBrief
            forecast={forecast}
            hitRate={hitRate}
            history={history}
            drivers={drivers}
          />
          <Disclaimer text={forecast.disclaimer} />
        </>
      )}
    </main>
  );
}
