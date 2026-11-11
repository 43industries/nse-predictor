"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Desk brief" },
  { href: "/market", label: "Market" },
  { href: "/banks", label: "Banks" },
  { href: "/health", label: "Health" },
];

export function TopBar() {
  const path = usePathname();
  return (
    <header className="topbar">
      <div className="brand">
        <p className="brand-wordmark">
          NSE <span>Predictor</span>
        </p>
        <span className="brand-tag">Trading companion</span>
      </div>
      <nav className="nav" aria-label="Primary">
        {LINKS.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link key={l.href} href={l.href} className={active ? "active" : undefined}>
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
