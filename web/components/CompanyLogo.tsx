"use client";

import { useState } from "react";
import { logoUrl } from "../lib/api";

type Props = {
  code: string;
  size?: "sm" | "lg";
};

export function CompanyLogo({ code, size = "sm" }: Props) {
  const ticker = String(code || "").toUpperCase();
  const [broken, setBroken] = useState(false);
  const sizeCls = size === "lg" ? " lg" : "";
  const letters = ticker.slice(0, 2) || "—";

  return (
    <span className={`logo-wrap${sizeCls}`} title={ticker}>
      {!broken && ticker && (
        <img
          className={`stock-logo${sizeCls}`}
          src={logoUrl(ticker)}
          alt=""
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={() => setBroken(true)}
        />
      )}
      <span
        className={`logo-mono${sizeCls}`}
        style={broken || !ticker ? undefined : { display: "none" }}
        aria-hidden={!broken}
      >
        {letters}
      </span>
    </span>
  );
}
