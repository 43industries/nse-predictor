import Link from "next/link";
import type { ReactNode } from "react";
import { stockPath } from "../lib/format";

type Props = {
  code: string;
  className?: string;
  children?: ReactNode;
};

export function StockLink({ code, className = "ticker-link", children }: Props) {
  const ticker = String(code || "").toUpperCase();
  if (!ticker) return <span>—</span>;
  return (
    <Link href={stockPath(ticker)} className={className}>
      {children ?? <strong>{ticker}</strong>}
    </Link>
  );
}
