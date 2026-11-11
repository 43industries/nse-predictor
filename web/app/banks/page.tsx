"use client";

import { useEffect, useState } from "react";
import { BanksTable } from "../../components/BanksTable";
import { Disclaimer } from "../../components/Disclaimer";
import { ApiError, api, type BankRow } from "../../lib/api";

export default function BanksPage() {
  const [rows, setRows] = useState<BankRow[]>([]);
  const [disclaimer, setDisclaimer] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const data = await api.banks();
        setRows(data.banks || []);
        setDisclaimer(data.disclaimer || "");
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Failed to load banks");
      }
    }
    load();
  }, []);

  return (
    <main>
      <p className="brand-tag">Kenya listed banks</p>
      <h1 className="hero-brand">
        FY <span>comparison</span>
      </h1>
      <p className="lede">
        Curated bank fundamentals (KES bn unless stated). Decision support only — not affiliated with
        kenyanbanks.com.
      </p>
      {error && <div className="error-banner">{error}</div>}
      <BanksTable rows={rows} />
      <Disclaimer text={disclaimer} />
    </main>
  );
}
