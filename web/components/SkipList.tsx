import type { SetupRow } from "../lib/api";
import { StockLink } from "./StockLink";

function joinedLinks(rows: SetupRow[], withReason = false) {
  return rows.map((s, i) => (
    <span key={s.stock_code}>
      {i > 0 ? ", " : ""}
      <StockLink code={s.stock_code} />
      {withReason ? ` (${s.skip_reason || "skip"})` : ""}
    </span>
  ));
}

export function SkipList({ skips }: { skips: SetupRow[] }) {
  const chance = skips.filter((s) =>
    ["at_chance", "thin_n"].includes(String(s.skip_reason || s.flag || ""))
  );
  const lowHit = skips.filter((s) => String(s.skip_reason || "") === "low_hit_rate");
  const other = skips.filter(
    (s) =>
      !["at_chance", "thin_n", "low_hit_rate"].includes(
        String(s.skip_reason || s.flag || "")
      )
  );
  return (
    <div>
      {chance.length > 0 && (
        <p className="muted">Skip / at-chance today: {joinedLinks(chance, true)}</p>
      )}
      {lowHit.length > 0 && (
        <p className="muted">Low hit rate: {joinedLinks(lowHit, true)}</p>
      )}
      {other.length > 0 && (
        <details>
          <summary className="muted">Other skips (flat / low conviction)</summary>
          <p className="muted">{joinedLinks(other, true)}</p>
        </details>
      )}
    </div>
  );
}
