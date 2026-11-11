import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { pct } from "../../lib/format";
import type { HealthView } from "../../lib/health";
import { KpiStrip } from "../KpiStrip";
import { RecordTable } from "../RecordTable";

export function ValidationPanel({ view }: { view: HealthView }) {
  const { gates, ric, ricb, cal, wfSum, fragile, abstain, sel, rq, proto } = view;

  return (
    <>
      {Object.keys(view.problem).length > 0 && (
        <p className="muted" style={{ marginTop: "1.25rem" }}>
          Direction = 5d excess vs NASI with ±{(view.deadZone * 100).toFixed(0)}% dead zone
          (dead_zone={pct(view.deadZone, 0)}). Return bands are 5d excess vs NASI.
        </p>
      )}

      <p className="section-label">Validation scorecard</p>
      <KpiStrip
        items={[
          { label: "Status", value: String(gates.status || "—").toUpperCase() },
          {
            label: "Test Rank IC",
            value: ric.mean_ic != null ? Number(ric.mean_ic).toFixed(3) : "—",
          },
          {
            label: "WF mean Rank IC",
            value: wfSum.wf_mean_rank_ic != null ? Number(wfSum.wf_mean_rank_ic).toFixed(3) : "—",
          },
          { label: "Abstain %", value: pct(abstain, 0) },
          { label: "ECE", value: cal.ece != null ? Number(cal.ece).toFixed(3) : "—" },
          { label: "Stress fragile", value: fragile ? "yes" : "no" },
        ]}
      />
      {Array.isArray(gates.reasons) && (gates.reasons as string[]).length > 0 && (
        <p className="muted">{(gates.reasons as string[]).join(" · ")}</p>
      )}
      {proto.embargo && (
        <p className="muted">
          Protocol: embargo {String(proto.horizon_days)}d · train_end {String(proto.train_end)} ·
          valid_end {String(proto.valid_end)}
        </p>
      )}
      {ric.n_days != null && (
        <p className="muted">
          Test Rank IC (prob vs excess return): mean{" "}
          <strong>{ric.mean_ic != null ? Number(ric.mean_ic).toFixed(3) : "—"}</strong>
          {ricb.ci_low != null && ricb.ci_high != null
            ? ` 95% bootstrap CI [${Number(ricb.ci_low).toFixed(3)}, ${Number(ricb.ci_high).toFixed(3)}]`
            : ""}{" "}
          over {String(ric.n_days)} days.
        </p>
      )}
      {view.atChance.length > 0 && (
        <p className="muted">
          At-chance names (near base-rate accuracy): {view.atChance.join(", ")}
        </p>
      )}

      {Object.keys(sel).length > 0 && (
        <>
          <p className="section-label">Validation selection</p>
          <KpiStrip
            items={[
              {
                label: "Direction threshold",
                value: sel.threshold != null ? Number(sel.threshold).toFixed(2) : "—",
              },
              { label: "Decision metric", value: String(sel.decision_metric || "—") },
              {
                label: "Calibration",
                value: String(sel.calibration || sel.calibration_status || "none"),
              },
              {
                label: "Return band (residual q)",
                value:
                  rq.q_low != null && rq.q_high != null
                    ? `${pct(rq.q_low as number, 1)} … ${pct(rq.q_high as number, 1)}`
                    : "—",
              },
            ]}
          />
          {view.selBits.length > 0 && <p className="muted">{view.selBits.join(" · ")}</p>}
        </>
      )}

      <p className="section-label">Baselines vs model</p>
      <div className="panel">
        <div className="chart-wrap">
          <ResponsiveContainer>
            <BarChart data={view.baselineChart}>
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#5c6b7a" }} />
              <YAxis
                domain={[0, "auto"]}
                tick={{ fontSize: 11, fill: "#5c6b7a" }}
                label={{
                  value: "Direction accuracy (%)",
                  angle: -90,
                  position: "insideLeft",
                  style: { fill: "#5c6b7a", fontSize: 11 },
                }}
              />
              <Tooltip formatter={(v: number) => [`${Number(v).toFixed(1)}%`, "Accuracy"]} />
              <Bar dataKey="acc" fill="#2f7d4a" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {view.wf.length > 0 && (
        <p className="muted">
          Walk-forward folds beating always-up baseline:{" "}
          <strong>
            {view.wfBeats}/{view.wf.length}
          </strong>
          . Prefer probabilities and ranges over point returns.
        </p>
      )}
      {view.cov.coverage != null && (
        <p className="muted">
          Return interval coverage on test: {pct(view.cov.coverage as number, 0)} (mean width{" "}
          {pct(view.cov.mean_width as number, 1)}).
        </p>
      )}
      {view.confusion.tp != null && (
        <p className="muted">
          Test confusion — TP {String(view.confusion.tp)} · FP {String(view.confusion.fp)} · TN{" "}
          {String(view.confusion.tn)} · FN {String(view.confusion.fn)}
        </p>
      )}

      {view.perStock.length > 0 && (
        <>
          <p className="section-label">Per-stock test accuracy</p>
          <RecordTable rows={view.perStock} maxHeight={280} />
        </>
      )}

      <details style={{ marginTop: "1.25rem" }}>
        <summary className="section-label" style={{ cursor: "pointer" }}>
          Validation detail
        </summary>
        <p className="muted">stress_fragile={fragile ? "true" : "false"} (OOS retrain)</p>
        {Object.keys(view.stress).length > 0 && (
          <div className="panel" style={{ overflowX: "auto", marginBottom: "0.75rem" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>period</th>
                  <th>mean_ic</th>
                  <th>n_days</th>
                  <th>abstain_rate</th>
                  <th>skipped</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(view.stress).map(([period, v]) => {
                  if (!v || typeof v !== "object") return null;
                  const row = v as Record<string, unknown>;
                  return (
                    <tr key={period}>
                      <td>{period}</td>
                      <td>{row.mean_ic != null ? Number(row.mean_ic).toFixed(3) : "—"}</td>
                      <td>{row.n_days == null ? "—" : String(row.n_days)}</td>
                      <td>{pct(row.abstain_rate as number, 0)}</td>
                      <td>{row.skipped ? "yes" : "no"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {view.calBins.length > 0 && (
          <div className="panel" style={{ marginBottom: "0.75rem" }}>
            <p className="section-label" style={{ marginTop: 0 }}>
              Calibration reliability (test, non-flat)
            </p>
            <div className="chart-wrap">
              <ResponsiveContainer>
                <LineChart data={view.calBins} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                  <CartesianGrid stroke="#e8eaed" />
                  <XAxis
                    dataKey="mean_prob"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: "#5c6b7a" }}
                    tickFormatter={(v) => Number(v).toFixed(1)}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: "#5c6b7a" }}
                    tickFormatter={(v) => Number(v).toFixed(1)}
                  />
                  <Tooltip
                    formatter={(v: number, name: string) => [
                      Number(v).toFixed(3),
                      name === "perfect" ? "Perfect" : "Frac up",
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="perfect"
                    stroke="#b0b7c3"
                    strokeDasharray="4 4"
                    dot={false}
                    name="perfect"
                  />
                  <Line
                    type="monotone"
                    dataKey="frac_up"
                    stroke="#2f7d4a"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="frac_up"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {view.wf.length > 0 && (
          <div className="panel" style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>fold</th>
                  <th>accuracy</th>
                  <th>rank_ic</th>
                  <th>abstain_rate</th>
                  <th>beats_baseline</th>
                </tr>
              </thead>
              <tbody>
                {view.wf.map((f, i) => {
                  const ricF = (f.rank_ic || {}) as Record<string, unknown>;
                  const m = (f.model || {}) as Record<string, unknown>;
                  return (
                    <tr key={String(f.fold ?? i)}>
                      <td>{String(f.fold ?? i)}</td>
                      <td>{pct(m.accuracy as number, 1)}</td>
                      <td>{ricF.mean_ic != null ? Number(ricF.mean_ic).toFixed(3) : "—"}</td>
                      <td>{pct(f.abstain_rate as number, 0)}</td>
                      <td>{f.beats_baseline ? "yes" : "no"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </details>

      <p className="section-label">Collection</p>
      <KpiStrip
        items={[
          { label: "Status", value: view.collection.status },
          { label: "Last run", value: view.collection.lastRun },
          { label: "Files", value: view.collection.files },
          { label: "Source", value: view.collection.source },
        ]}
      />
    </>
  );
}
