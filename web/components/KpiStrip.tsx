type Item = { label: string; value: string };

export function KpiStrip({ items }: { items: Item[] }) {
  return (
    <div className="kpi-strip">
      {items.map((it) => (
        <div className="kpi" key={it.label}>
          <div className="k">{it.label}</div>
          <div className="v">{it.value}</div>
        </div>
      ))}
    </div>
  );
}
