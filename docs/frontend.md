# Frontend

The Command Center is implemented in `src/frontend/web/` as three plain files: `index.html`, `styles.css`, `app.js`. No framework, no build step. The only runtime dependency is MapLibre GL JS, loaded from a CDN.

## Run

It is served by the same FastAPI process as the API:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`.

## Data Source

The entire frontend loads one payload once at boot:

- `GET /dashboard`, which serves `data/results/dashboard.json`

If missing, generate it:

```powershell
python -m src.features.build_road_risk --city "Bengaluru,Mumbai,Hyderabad" --max-roads 1500
```

Everything interactive afterward happens client-side against that payload; there are no further server round-trips during normal use.

## Key UX Sections

- **Command Center (overview):** a real dark basemap with every analyzed road drawn as a colored corridor (critical, watch, stable). A left panel shows the city risk snapshot in plain language and an "Act First" list of the highest-priority named roads.
- **Road drill-down:** click any road to open a detail drawer with its priority score, city rank, months-to-critical countdown, top stress drivers, a trend sparkline against the critical band, and a recommended action.
- **Time Machine:** replays 2020 to 2024 month by month. Roads re-color live as their condition changes, three driver meters (rainfall, standing water, heat) move with the data, and a narrated caption explains what is happening and where.
- **Budget Planner:** a what-if slider that greedily builds a prioritized work order from the highest-risk roads and reports roads protected, kilometres resurfaced early, and estimated reactive cost avoided.
- **Executive Brief:** one click generates a print-ready report (KPIs, recommended investment, top 20 roads, a six-month action plan).

## Filtering and Decision Flow

Top bar controls:
- City switcher (Bengaluru, Mumbai, Hyderabad)
- Mode switcher (Command Center, Time Machine, Budget Planner)
- Road search, ranked by risk

Typical workflow:
1. Start on the Command Center: read the city snapshot, scan the "Act First" list.
2. Drill into specific roads for drivers and recommended actions.
3. Switch to Time Machine to see how the risk built up over time.
4. Switch to Budget Planner to turn a budget number into a work order.
5. Export the Executive Brief for a print-ready summary.

## Visual Semantics

Risk tiers are **relative percentiles within each city**, not fixed absolute thresholds:
- `High` (critical, red): top 15% of roads by risk score in that city
- `Medium` (watch, amber): next 35%
- `Low` (stable, slate): the rest

The cut points for the currently loaded city are in the dashboard payload's `city_cuts` object and are recomputed every time the pipeline reruns.

## UX Troubleshooting

- Splash screen never dismisses, or map never appears:
  - open the browser console; a MapLibre or `/dashboard` fetch error will show there
  - confirm `data/results/dashboard.json` exists and is not empty
- Map shows roads from only one part of the city:
  - regenerate the road ranking from the latest risk scores
  - verify a mix of High/Medium/Low roads exists in `road_risk_ranking.json`
- Stale content after a code change:
  - `index.html` versions `app.js`/`styles.css` with a query string (`?v=N`) for cache busting; bump it after editing either file
