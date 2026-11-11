import type { BankRow } from "../lib/api";
import { num } from "../lib/format";
import { CompanyLogo } from "./CompanyLogo";
import { StockLink } from "./StockLink";

export function BanksTable({ rows }: { rows: BankRow[] }) {
  if (!rows.length) {
    return <p className="muted">No bank fundamentals on file.</p>;
  }
  return (
    <div className="panel" style={{ overflowX: "auto" }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Bank</th>
            <th>Assets</th>
            <th>Loans</th>
            <th>Deposits</th>
            <th>PAT</th>
            <th>PAT YoY</th>
            <th>ROE</th>
            <th>NIM</th>
            <th>NPL</th>
            <th>EPS</th>
            <th>DPS</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const yoy = r.pat_yoy;
            const yoyCls =
              yoy == null || Number.isNaN(Number(yoy))
                ? ""
                : Number(yoy) > 0
                  ? "up"
                  : Number(yoy) < 0
                    ? "down"
                    : "";
            return (
              <tr key={r.stock_code}>
                <td>
                  <StockLink code={r.stock_code} className="ticker-cell">
                    <CompanyLogo code={r.stock_code} />
                    <span>
                      <strong>{r.bank_name}</strong>{" "}
                      <span className="muted">({r.stock_code})</span>
                    </span>
                  </StockLink>
                </td>
                <td>{num(r.assets, 0)}</td>
                <td>{num(r.loans, 0)}</td>
                <td>{num(r.deposits, 0)}</td>
                <td>{num(r.pat, 1)}</td>
                <td className={yoyCls}>
                  {yoy == null ? "—" : `${Number(yoy) > 0 ? "+" : ""}${num(yoy, 1)}%`}
                </td>
                <td>{r.roe == null ? "—" : `${num(r.roe, 1)}%`}</td>
                <td>{r.nim == null ? "—" : `${num(r.nim, 1)}%`}</td>
                <td>{r.npl == null ? "—" : `${num(r.npl, 1)}%`}</td>
                <td>{num(r.eps, 2)}</td>
                <td>{num(r.dividend, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
