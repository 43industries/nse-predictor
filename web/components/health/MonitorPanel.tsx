import { fmtChecked, pct } from "../../lib/format";
import type { HealthView } from "../../lib/health";
import { KpiStrip } from "../KpiStrip";
import { RecordTable } from "../RecordTable";

export function MonitorPanel({
  view,
  history,
}: {
  view: HealthView;
  history: Record<string, unknown>[];
}) {
  if (!view.hasMonitor) {
    return (
      <>
        <p className="section-label">Monitor</p>
        <p className="muted">
          No monitor_status.json yet. Run <code>python scripts/run_monitor.py</code> or the daily
          job.
        </p>
      </>
    );
  }

  return (
    <>
      <p className="section-label">Monitor</p>
      <KpiStrip
        items={[
          { label: "Status", value: view.monitorStatus },
          { label: "Forecast as_of", value: String(view.score.as_of || "—") },
          { label: "Abstain %", value: pct(view.score.abstain_rate as number, 0) },
          { label: "Lagged hit-rate", value: pct(view.laggedHit, 0) },
          {
            label: "Val gates",
            value: String(view.gatesM.validation_status || "—").toUpperCase(),
          },
        ]}
      />
      {view.checkedAt != null && (
        <p className="muted">Last check: {fmtChecked(view.checkedAt)}Z</p>
      )}
      {view.alerts.length > 0 ? (
        <div className="panel" style={{ overflowX: "auto", marginTop: "0.75rem" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Code</th>
                <th>Severity</th>
                <th>Message</th>
                <th>At</th>
              </tr>
            </thead>
            <tbody>
              {view.alerts.map((a, i) => (
                <tr key={`${a.code}-${i}`}>
                  <td>{String(a.code || "—")}</td>
                  <td>{String(a.severity || "—").toUpperCase()}</td>
                  <td>{String(a.message || "—")}</td>
                  <td>{fmtChecked(a.at) || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No alerts on last run.</p>
      )}
      {history.length > 0 && (
        <>
          <p className="section-label">Recent monitor history</p>
          <RecordTable rows={history} maxCols={8} />
        </>
      )}
    </>
  );
}
