# ER diagram notes (PRD Section 14)

Suggested tables (logical):
- `customers` (customer_id, first_seen, last_seen, country, ...)
- `transactions` (invoice, stock_code, customer_id, quantity, price, invoice_date, ...)
- `predictions` (customer_id, churn_probability, churn_label, model_version, scored_at, ...)
- `segments` (customer_id, segment_label, method, scored_at, ...)

Export an ER diagram image to `docs/er_diagram.png` when ready.
