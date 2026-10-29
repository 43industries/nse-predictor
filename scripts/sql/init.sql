-- NSE Predictor schema (Postgres)
CREATE TABLE IF NOT EXISTS prices (
    trade_date DATE NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    low_12m DOUBLE PRECISION,
    high_12m DOUBLE PRECISION,
    day_low DOUBLE PRECISION,
    day_high DOUBLE PRECISION,
    day_close DOUBLE PRECISION,
    prev_price DOUBLE PRECISION,
    change_value DOUBLE PRECISION,
    change_pct DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    adj_price DOUBLE PRECISION,
    source_file TEXT,
    PRIMARY KEY (trade_date, stock_code)
);

CREATE INDEX IF NOT EXISTS idx_prices_code_date ON prices (stock_code, trade_date);

CREATE TABLE IF NOT EXISTS sectors (
    stock_code TEXT PRIMARY KEY,
    stock_name TEXT,
    market_sector TEXT,
    as_of_year INT
);

CREATE TABLE IF NOT EXISTS macro (
    observe_date DATE PRIMARY KEY,
    cbk_rate DOUBLE PRECISION,
    tbill_91d DOUBLE PRECISION,
    usd_kes DOUBLE PRECISION,
    inflation_yoy DOUBLE PRECISION,
    lending_rate DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS fundamentals (
    stock_code TEXT NOT NULL,
    report_date DATE NOT NULL,
    eps DOUBLE PRECISION,
    dividend DOUBLE PRECISION,
    book_value DOUBLE PRECISION,
    period TEXT,
    PRIMARY KEY (stock_code, report_date)
);

CREATE TABLE IF NOT EXISTS forecasts (
    as_of_date DATE NOT NULL,
    stock_code TEXT NOT NULL,
    horizon_days INT NOT NULL,
    target_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    pred_label TEXT,
    prob_up DOUBLE PRECISION,
    pred_value DOUBLE PRECISION,
    lower DOUBLE PRECISION,
    upper DOUBLE PRECISION,
    metrics_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, stock_code, horizon_days, target_type, model_name)
);
