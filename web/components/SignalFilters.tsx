"use client";

import type { ReactNode } from "react";

type Props = {
  minProb: number;
  minHit?: number;
  showHit?: boolean;
  onMinProb: (v: number) => void;
  onMinHit?: (v: number) => void;
  onRefresh?: () => void;
  children?: ReactNode;
};

export function SignalFilters({
  minProb,
  minHit = 0.52,
  showHit = true,
  onMinProb,
  onMinHit,
  onRefresh,
  children,
}: Props) {
  return (
    <div className="panel form-grid filters-row">
      <p className="section-label" style={{ marginTop: 0, gridColumn: "1 / -1" }}>
        Filters
      </p>
      <label>
        Min P(up)
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
          Min hit rate
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
