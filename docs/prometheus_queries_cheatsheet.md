# Prometheus Query Cheatsheet (Demo Reference)

Copy-paste into Prometheus's **Graph** tab (`http://localhost:9090`) or a Grafana
panel's query box (`http://localhost:3000`). Ordered for a live walkthrough:
health → traffic → latency → errors → model confidence → drift.

Before running 5-7, make a few real `/predict` calls from Streamlit first —
those metrics stay at `0`/`NaN` until at least one prediction has been made.

## 1. Is the service healthy?
```promql
up{job=~"api-.*"}
```
`1` = API is up and being scraped.

## 2. How much traffic is it getting?
```promql
handler:http_requests:rate5m
```
Requests/sec per endpoint (precomputed recording rule).

## 3. Latency — p95
```promql
handler:http_request_duration_seconds:p95
```
95% of requests finish faster than this, per endpoint.

## 4. Error rate
```promql
job:http_requests_error_ratio:rate5m
```
Share of requests that returned 5xx (0 = no errors).

## 5. Model prediction confidence
```promql
job:prediction_confidence:p50
```
Median top-1 confidence score. Low value → model is unsure on a lot of products.

## 6. Prediction drift — PSI (main talking point)
```promql
prediction_drift_psi
```
`<0.1` no drift · `0.1-0.25` moderate · `>0.25` significant.
Grafana alert fires if this stays `>0.60` for 2 consecutive minutes.

## 7. Drift window fill
```promql
prediction_drift_window_size
```
How many of the last predictions (max 200) are feeding the PSI calculation above.

## Bonus — resource usage
```promql
process_resident_memory_bytes{job=~"api-.*"}
```

---

### Where these live in the repo
- Recording rules (`handler:...`, `job:...`): `monitoring/prometheus/rules.yml`
- Grafana dashboard panels: `monitoring/grafana/dashboards/api-overview.json`
- Drift alert rule: `monitoring/grafana/provisioning/alerting/rules.yaml`
- Metrics are defined/emitted in: `src/api/main.py` (custom) +
  `prometheus-fastapi-instrumentator` (HTTP metrics, automatic)
