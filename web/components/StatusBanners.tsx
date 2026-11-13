import Link from "next/link";

export function StatusBanners({
  error,
  msg,
  portfolioLink,
}: {
  error?: string;
  msg?: string;
  portfolioLink?: boolean;
}) {
  return (
    <>
      {error ? <div className="error-banner">{error}</div> : null}
      {msg && !error ? (
        <p className="ok-banner">
          {msg}
          {portfolioLink ? (
            <>
              {" "}
              <Link href="/portfolio" className="ticker-link">
                Open portfolio →
              </Link>
            </>
          ) : null}
        </p>
      ) : null}
    </>
  );
}
