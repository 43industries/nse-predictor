import type { ReactNode } from "react";
import { Fraunces, Manrope } from "next/font/google";
import { TickerBar } from "../components/TickerBar";
import { TopBar } from "../components/TopBar";
import "./globals.css";

const sans = Manrope({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

export const metadata = {
  title: "NSE Predictor",
  description: "Decision-support trading companion for the Nairobi Securities Exchange",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${display.variable}`}>
      <body>
        <div className="app-shell">
          <TopBar />
          <TickerBar />
          <div className="page">{children}</div>
        </div>
      </body>
    </html>
  );
}
