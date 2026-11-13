import { api, type Forecast, type SetupRow } from "./api";

/** Create a pending paper ticket from a desk/market suggestion row. */
export async function ticketFromSuggestion(
  row: Pick<
    SetupRow & Forecast,
    "stock_code" | "suggested_side" | "suggested_shares" | "suggest_note" | "book_status"
  >,
  source: string
) {
  if (!row.suggested_side || !(row.suggested_shares || 0)) {
    throw new Error("No suggested size for this name");
  }
  return api.portfolioCreateTicket({
    stock_code: String(row.stock_code).toUpperCase(),
    side: row.suggested_side,
    shares: Number(row.suggested_shares),
    reason: `${source}:${row.suggest_note || row.book_status || "setup"}`,
    suggestion_action: row.suggest_note || source,
  });
}
