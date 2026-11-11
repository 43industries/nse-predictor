import { cellText } from "../lib/format";
import { StockLink } from "./StockLink";

type Props = {
  rows: Record<string, unknown>[];
  columns?: string[];
  maxCols?: number;
  stockCodeKey?: string;
  className?: string;
  maxHeight?: number | string;
};

export function RecordTable({
  rows,
  columns,
  maxCols,
  stockCodeKey = "stock_code",
  className,
  maxHeight,
}: Props) {
  if (!rows.length) return null;
  let cols = columns || Object.keys(rows[0]);
  if (maxCols != null) cols = cols.slice(0, maxCols);

  return (
    <div className={className || "panel"} style={{ overflowX: "auto", maxHeight }}>
      <table className="data-table">
        <thead>
          <tr>
            {cols.map((k) => (
              <th key={k}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={String(row[stockCodeKey] ?? i)}>
              {cols.map((k) => (
                <td key={k}>
                  {k === stockCodeKey && row[k] != null ? (
                    <StockLink code={String(row[k])} />
                  ) : (
                    cellText(row[k])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
