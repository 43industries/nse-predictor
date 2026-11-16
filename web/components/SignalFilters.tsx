"use client";

import type { ReactNode } from "react";

export type FilterPreset = "strict" | "normal" | "loose" | "custom";

export const FILTER_PRESETS: Record<
  Exclude<FilterPreset, "custom">,
  { minProb: number; minHit: number; label: string }
> = {
  strict: { minProb: 0.6, minHit: 0.55, label: "Strict" },
  normal: { minProb: 0.55, minHit: 0.52, label: "Normal" },
  loose: { minProb: 0.5, minHit: 0.5, label: "Loose" },
};

export function matchFilterPreset(minProb: number, minHit: number): FilterPreset {
  for (const [key, p] of Object.entries(FILTER_PRESETS) as Array<
    [Exclude<FilterPreset, "custom">, (typeof FILTER_PRESETS)["normal"]]
  >) {
    if (Math.abs(p.minProb - minProb) < 1e-9 && Math.abs(p.minHit - minHit) < 1e-9) {
      return key;
    }
  }
  return "custom";
}

type Props = {
  minProb: number;
  minHit?: number;
  showHit?: boolean;
  showPresets?: boolean;
  onMinProb: (v: number) => void;
  onMinHit?: (v: number) => void;
  onPreset?: (preset: Exclude<FilterPreset, "custom">) => void;
  onRefresh?: () => void;
  children?: ReactNode;
};

export function SignalFilters({
  minProb,
  minHit = 0.52,
  showHit = true,
  showPresets = false,
  onMinProb,
  onMinHit,
  onPreset,
  onRefresh,
  children,
}: Props) {
  const active = matchFilterPreset(minProb, minHit);

  return (
    <div className="panel form-grid filters-row">
      <p className="section-label" style={{ marginTop: 0, gridColumn: "1 / -1" }}>
        Filters
      </p>
      {showPresets && onPreset ? (
        <div className="preset-row" style={{ gridColumn: "1 / -1" }}>
          {(Object.keys(FILTER_PRESETS) as Array<Exclude<FilterPreset, "custom">>).map(
            (key) => (
              <button
                key={key}
                type="button"
                className={`btn preset-btn${active === key ? " preset-btn-active" : ""}`}
                onClick={() => onPreset(key)}
              >
                {FILTER_PRESETS[key].label}
              </button>
            )
          )}
          {active === "custom" ? <span className="muted small">Custom</span> : null}
        </div>
      ) : null}
      <label>
        Stronger signals (min chance up)
        <input
          type="number"
          min={0}
          max={0.9}
          step={0.01}
          value={minProb}
          onChange={(e) => onMinProb(Number(e.target.value))}
        />
      </label>
      {showHit && onMinHit && (
        <label>
          Min recent accuracy
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={minHit}
            onChange={(e) => onMinHit(Number(e.target.value))}
          />
        </label>
      )}
      {children}
      {onRefresh && (
        <button className="btn" type="button" onClick={onRefresh}>
          Refresh
        </button>
      )}
    </div>
  );
}
