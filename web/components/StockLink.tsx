import Link from "next/link";
import type { ReactNode } from "react";
import { stockPath } from "../lib/format";

type Props = {
  code: string;
  className?: string;
  title?: string;
  children?: ReactNode;
};

export function StockLink({ code, className = "ticker-link", title, children }: Props) {
  const ticker = String(code || "").toUpperCase();
  if (!ticker) return <span>—</span>;
  return (
    <Link href={stockPath(ticker)} className={className} title={title}>
      {children ?? <strong>{ticker}</strong>}
    </Link>
  );
}
