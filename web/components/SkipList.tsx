import type { SetupRow } from "../lib/api";
import { StockLink } from "./StockLink";

type Group = {
  title: string;
  blurb: string;
  rows: SetupRow[];
};

function GroupPanel({ title, blurb, rows }: Group) {
  if (!rows.length) return null;
  const shown = rows.slice(0, 4);
  const more = rows.length - shown.length;
  return (
    <div className="skip-group panel">
      <p className="skip-group-title">
        {title} <span className="muted">({rows.length})</span>
      </p>
      <p className="muted small">{blurb}</p>
      <p className="skip-group-tickers">
        {shown.map((s, i) => (
          <span key={s.stock_code}>
            {i > 0 ? ", " : ""}
            <StockLink
              code={s.stock_code}
              title={s.skip_label || s.skip_reason || undefined}
            />
          </span>
        ))}
        {more > 0 ? <span className="muted"> +{more} more</span> : null}
      </p>
    </div>
  );
}

export function SkipList({ skips }: { skips: SetupRow[] }) {
  if (!skips.length) {
    return <p className="muted">No skips — every name cleared the filters.</p>;
  }

  const unreliable = skips.filter((s) =>
    ["at_chance", "thin_n"].includes(String(s.skip_reason || s.flag || ""))
  );
  const tooWeak = skips.filter((s) =>
    ["low_conviction", "low_hit_rate"].includes(String(s.skip_reason || ""))
  );
  const sittingOut = skips.filter((s) => {
    const reason = String(s.skip_reason || s.flag || "");
    return !["at_chance", "thin_n", "low_conviction", "low_hit_rate"].includes(reason);
  });

  return (
    <div className="skip-groups">
      <GroupPanel
        title="Unreliable"
        blurb="Model no better than a coin-flip on these names lately."
        rows={unreliable}
      />
      <GroupPanel
        title="Too weak"
        blurb="Below the confidence bar or recent accuracy floor."
        rows={tooWeak}
      />
      <GroupPanel
        title="Sitting out"
        blurb="Flat calls and other non-actionable names."
        rows={sittingOut}
      />
    </div>
  );
}
