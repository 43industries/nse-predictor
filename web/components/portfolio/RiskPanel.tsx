import type { PortfolioRisk } from "../../lib/api";
import { kes, pct } from "../../lib/format";

export function RiskPanel({ risk }: { risk?: PortfolioRisk | null }) {
  if (!risk) return null;
  const pol = risk.policy || {};
  const util = risk.utilization || {};
  const breaches = risk.breaches || [];
  return (
    <div className="panel risk-panel">
      <p className="section-label" style={{ marginTop: 0 }}>
        Risk
      </p>
      <div className="kpi-strip">
        <div className="kpi">
          <div className="k">Names</div>
          <div className="v">
            {util.n_names ?? 0}/{pol.max_names ?? "—"}
          </div>
        </div>
        <div className="kpi">
          <div className="k">Cash %</div>
          <div className="v">{pct(util.cash_pct, 0)}</div>
        </div>
        <div className="kpi">
          <div className="k">Min cash</div>
          <div className="v">{pct(pol.min_cash_pct, 0)}</div>
        </div>
        <div className="kpi">
          <div className="k">Max wt</div>
          <div className="v">
            {pct(util.max_position_weight, 0)} / {pct(pol.max_weight_pct, 0)}
          </div>
        </div>
        <div className="kpi">
          <div className="k">Lot / bps</div>
          <div className="v">
            {pol.lot_size ?? "—"} / {pol.cost_bps ?? "—"}
          </div>
        </div>
        <div className="kpi">
          <div className="k">NAV</div>
          <div className="v">{kes(util.nav_kes)}</div>
        </div>
      </div>
      {breaches.length > 0 ? (
        <ul className="breach-list">
          {breaches.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
      ) : (
        <p className="muted small">No policy breaches.</p>
      )}
    </div>
  );
}
