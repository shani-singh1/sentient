# Frontend

Dashboard is implemented in `src/frontend/app.py` using Streamlit, Plotly, and PyDeck.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frontend.ps1
```

Alternative:

```powershell
python -m streamlit run src/frontend/app.py --server.port 8501
```

Default URL: `http://localhost:8501`

## Data Source

Primary input:
- `data/results/road_risk_ranking.json`

If missing, generate it:

```powershell
python -m src.features.build_road_risk --max-roads 300
```

## Key UX Sections

- **Executive summary:** high/medium/low counts and at-risk road length.
- **Risk overview charts:**
  - risk distribution donut
  - top risky roads bar chart
  - zone-wise stack chart
  - average risk by road type
- **Risk map:** pydeck path layer with red/amber/green road colors.
- **Ranking table:** sorted road list with risk and attributes.

## Filtering and Decision Flow

Sidebar controls:
- Top N roads
- Risk level filter (`High`, `Medium`, `Low`)

Typical workflow:
1. Start with executive summary and top risky roads.
2. Use map to inspect where risks cluster.
3. Export or capture top entries for field inspection planning.

## Visual Semantics

- `High` risk: `risk_score >= 0.7` (red)
- `Medium`: `0.4 <= risk_score < 0.7` (amber)
- `Low`: `< 0.4` (green)

## UX Troubleshooting

- Map shows only one area:
  - regenerate road ranking using latest risk scores
  - verify mixed risk levels exist in output JSON
- Table looks unreadable:
  - clear browser cache
  - restart Streamlit to pick latest CSS updates
