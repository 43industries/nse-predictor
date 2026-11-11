export function Disclaimer({ text }: { text?: string }) {
  return (
    <p className="disclaimer">
      {text ||
        "Forecasts are for internal research and decision support only. They are not investment advice."}
    </p>
  );
}
