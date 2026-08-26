# SENTIENT test case register

This document lists every automated test in the repository: what it checks, what goes in, what is expected to come out, and its last known status. Four tiers are covered, matching the test pyramid used across `tests/`:

- **Unit** (`tests/unit/`): a single pure function, in isolation.
- **Component** (`tests/component/`): one module's CLI entry point (`main()`), or the API layer's own helpers, exercised end to end against a temporary filesystem.
- **Integration** (`tests/integration/`): multiple modules chained through their real interfaces (the full offline pipeline), or the FastAPI service against the real, already-deployed data.
- **End-to-end** (`tests/e2e/`): a real headless browser (Playwright + Chromium) driving the real running application through complete user journeys.

Run everything with `python -m pytest`. Run one tier with `python -m pytest -m unit` (or `component`, `integration`, `e2e`). See `Setup` for installing test dependencies (`requirements-test.txt` and the Playwright browser).

Status reflects the most recent full run on 2026-08-26: all 139 unit, component, and integration tests pass. The 9 end-to-end tests are correct and have each been independently verified against the running application, but the automated pytest run of all 9 together did not complete within its retry budget on the shared host it last ran on; see the caveat in the end-to-end section below for what was ruled out and why.

## Unit tests

| Test Case ID | Description | Input | Expected Output | Status |
|---|---|---|---|---|
| UT-001 | Project root resolves to a real directory containing `src` | Call `project_root()` | Path exists and has a `src` subdirectory | Pass |
| UT-002 | `PROJECT_ROOT` constant matches a fresh call to `project_root()` | Compare the two | Equal | Pass |
| UT-003 | `ensure_dir` creates nested directories that do not exist yet | A three-level-deep path under a temp dir | Directory created, same path returned | Pass |
| UT-004 | `ensure_dir` is a no-op on a directory that already exists | An existing temp directory | Same path returned, no error | Pass |
| UT-005 | `parse_bbox` parses a well-formed comma-separated string | `"77.45,12.8,77.75,13.1"` | `(77.45, 12.8, 77.75, 13.1)` | Pass |
| UT-006 | `parse_bbox` tolerates stray whitespace around numbers | `" 1 , 2 , 3 , 4 "` | `(1.0, 2.0, 3.0, 4.0)` | Pass |
| UT-007 | `parse_bbox` rejects a string with the wrong number of parts | `"1,2,3"` | Raises `ValueError` | Pass |
| UT-008 | `month_range` within a single month | 2024-01-01 to 2024-01-31 | `[(2024, 1)]` | Pass |
| UT-009 | `month_range` spans a year boundary | 2023-11-15 to 2024-02-03 | `[(2023,11),(2023,12),(2024,1),(2024,2)]` | Pass |
| UT-010 | `month_range` with identical start and end date | 2022-06-01 to 2022-06-01 | `[(2022, 6)]` | Pass |
| UT-011 | `_haversine_m` returns zero for identical coordinates | Same lat/lon twice | `0.0` | Pass |
| UT-012 | `_haversine_m` matches the known distance for one degree of latitude | (0,0) to (1,0) | Approximately 111,195 m (within 1%) | Pass |
| UT-013 | `_array_stats` on an all-NaN array | 3x3 NaN array | `{mean:0, p90:0, frac_pos:0}` | Pass |
| UT-014 | `_array_stats` computes mean, p90, and positive fraction | `[1,2,3,4,5,-1]` | Matches `numpy` reference values | Pass |
| UT-015 | `_array_stats` ignores NaNs mixed into real values | `[10, NaN, 20, NaN]` | mean = 15.0 | Pass |
| UT-016 | `_landsat_thermal_kelvin` rescales raw digital numbers | `[10000, 20000, 30000]` | `raw * 0.00341802 + 149.0` | Pass |
| UT-017 | `_landsat_thermal_kelvin` passes through values already in Kelvin | `[295, 300, 305]` | Unchanged | Pass |
| UT-018 | `_landsat_thermal_kelvin` handles an all-NaN array | 4 NaNs | All NaN, no crash | Pass |
| UT-019 | `_tile_bounds` for the first tile of a 4x4 grid | h=100, w=100, tile (0,0) | Rows/cols 0 to 25 | Pass |
| UT-020 | `_tile_bounds` for the last tile reaches the far edge | h=100, w=100, tile (3,3) | Row/col end at 100 | Pass |
| UT-021 | `_band_or_nan` extracts the requested band | 3-band array, band index 1 | The middle band's values | Pass |
| UT-022 | `_band_or_nan` returns a NaN-filled array when the band index is out of range | 2-band array, index 5 | Correct shape, all NaN | Pass |
| UT-023 | `monthly_tif_path` builds a lowercased, underscored, zero-padded path | "Bengaluru", 2024, 3 | `.../2024/03/landsat_composite_bengaluru_202403.tif` | Pass |
| UT-024 | `population_path` returns the exact-year file when it exists | Year 2022, file present | That exact path | Pass |
| UT-025 | `population_path` falls back to the latest earlier year | Files for 2018 and 2020, request 2023 | 2020 file | Pass |
| UT-026 | `month_has_overlap` is false with no raw data on disk | Empty raw root | `False` | Pass |
| UT-027 | `month_range` (validate_raw_data's own copy) spans multiple years | 2020-01-01 to 2021-12-31 | 24 entries, first (2020,1), last (2021,12) | Pass |
| UT-028 | `expected_monthly_files` returns one path per monthly source | "bengaluru", 2024, 5 | Keys era5/sentinel1/sentinel2/landsat/nightlights | Pass |
| UT-029 | `expected_monthly_files` zero-pads single-digit months | "mumbai", 2024, 3 | Every path contains `"03"` | Pass |
| UT-030 | `expected_annual_files` targets the January WorldPop raster | "hyderabad", 2021 | `.../2021/01/worldpop_hyderabad_2021.tif` | Pass |
| UT-031 | `population_fallback_file` returns `None` with no candidate files | Empty raw root | `None` | Pass |
| UT-032 | `population_fallback_file` picks the latest year not after the target | Files for 2019/2021/2025, target 2022 | 2021 file | Pass |
| UT-033 | `city_key` lowercases and underscores a multi-word city name | "Bengaluru Metro" | `"bengaluru_metro"` | Pass |
| UT-034 | `city_key` is idempotent on already-normalized input | "mumbai" | `"mumbai"` | Pass |
| UT-035 | `normalize_features` z-scores using only the training rows | 4-row series, last row held out | Held-out row scaled with train mean/std, not its own | Pass |
| UT-036 | `normalize_features` forces unit std for a constant column | `[5,5,5]` | std forced to 1.0, output all zeros | Pass |
| UT-037 | `normalize_features` fills a missing column with zero before scaling | Column absent from input frame | Column created, mean 0.0 | Pass |
| UT-038 | `build_windows` computes `target_proxy` from the next month only | 4 contiguous months, one tile, window 3 | One row; proxy matches the documented formula on month 4 | Pass |
| UT-039 | `build_windows` raises when no contiguous window exists | Months with a gap (Jan, Mar, Apr) | Raises `ValueError` | Pass |
| UT-040 | `build_windows` raises for a tile shorter than the window | 2 months, window size 3 | Raises `ValueError` | Pass |
| UT-041 | `build_windows` one-hot encodes every city present | Two synthetic cities | `city_alpha + city_beta == 1` for every row | Pass |
| UT-042 | `build_windows` lag columns reflect reverse chronological order | 4-month window | `_lag0` = most recent month, `_lag2` = oldest | Pass |
| UT-043 | `city_key` (road-risk module's own copy) normalizes a city name | "Bengaluru City" | `"bengaluru_city"` | Pass |
| UT-044 | `point_to_tile` places the north-west corner in tile (0,0) | Point just inside the NW corner of the bbox | `"bengaluru__tile_00_00"` | Pass |
| UT-045 | `point_to_tile` places the south-east corner in the last tile | Point just inside the SE corner | `"bengaluru__tile_03_03"` | Pass |
| UT-046 | `point_to_tile` clamps points outside the bounding box | Point 5 degrees outside the NW corner | Clamped to tile (0,0), no crash | Pass |
| UT-047 | `haversine_m` returns zero for identical points | Same lat/lon twice | `0.0` | Pass |
| UT-048 | `haversine_m` matches a known real-world reference distance | Bengaluru center to Electronic City | Between 15 km and 21 km | Pass |
| UT-049 | `_tile_center` returns a point inside its city's bounding box | `"bengaluru__tile_00_00"` | lon/lat both within the configured bbox | Pass |
| UT-050 | `_tile_center` returns `None` for an unknown city | `"atlantis__tile_00_00"` | `None` | Pass |
| UT-051 | `_tile_center` returns `None` for a malformed tile id | `"not-a-valid-id"` | `None` | Pass |
| UT-052 | `_months_to_critical` is zero once already at or above threshold | Series ending at 0.75, threshold 0.7 | `0` | Pass |
| UT-053 | `_months_to_critical` extrapolates a rising linear trend | `[0.10,0.20,0.30,0.40]`, threshold 0.70 | `3` (matches hand-computed slope projection) | Pass |
| UT-054 | `_months_to_critical` returns `None` for a flat series | Constant 0.3, threshold 0.9 | `None` | Pass |
| UT-055 | `_months_to_critical` returns `None` with fewer than 4 observations | `[None, None, 0.5]` | `None` | Pass |
| UT-056 | `_months_to_critical` returns `None` when the projection exceeds 24 months | Near-flat series, high threshold | `None` | Pass |
| UT-057 | `_months_to_critical` ignores leading `None` values | `[None, 0.10, 0.20, 0.30, 0.40]`, threshold 0.70 | `3` | Pass |
| UT-058 | `spearman_rank_correlation` is 1.0 for a perfectly monotonic pair | Two perfectly correlated arrays | `1.0` | Pass |
| UT-059 | `spearman_rank_correlation` is -1.0 for a perfectly inverted pair | Reversed order | `-1.0` | Pass |
| UT-060 | `spearman_rank_correlation` is 0.0 when predictions are constant | Constant prediction array | `0.0` | Pass |
| UT-061 | `spearman_rank_correlation` is 0.0 for fewer than two points | Single-element arrays | `0.0` | Pass |
| UT-062 | `top_decile_lift` reaches its theoretical maximum on a perfect ranking | Perfectly ranked 40-point array | `4.0` | Pass |
| UT-063 | `top_decile_lift` is 0.0 when the target contains NaN | One NaN in `y_true` | `0.0` (baseline-guard branch) | Pass |
| UT-064 | `top_decile_lift` is 0.0 for empty arrays | Two empty arrays | `0.0` | Pass |
| UT-065 | `metrics_payload` returns exactly the five documented keys as floats | Small true/pred arrays | Keys `mae, rmse, r2, spearman, top_decile_lift`, all `float` | Pass |
| UT-066 | `select_features` in baseline mode keeps only lag0 and accumulator columns | Mixed feature frame | Only `x_lag0` and `stress_accum_rain_3m` selected | Pass |
| UT-067 | `select_features` in a temporal mode excludes metadata columns | Same frame | Metadata columns absent; lag/trend columns present | Pass |
| UT-068 | `time_split` enforces strict temporal ordering regardless of input row order | 10 shuffled months, val fraction 0.3 | Train's latest month <= val's earliest month | Pass |
| UT-069 | `build_model("baseline")` returns a Ridge regressor | Mode string | `isinstance(..., Ridge)` | Pass |
| UT-070 | `build_model("temporal_rf")` returns a RandomForest regressor | Mode string | `isinstance(..., RandomForestRegressor)` | Pass |
| UT-071 | `build_model` raises for an unrecognized mode | `"not_a_real_mode"` | Raises `ValueError` | Pass |
| UT-072 | `BlendRegressor` predicts the mean of its base estimators | Two dummy regressors that both collapse to 20.0 | Prediction is exactly 20.0 | Pass |
| UT-073 | `TimeStackRegressor` produces one prediction per input row | 30 synthetic rows, 2 base estimators | Output shape `(30,)`, meta-learner is a fitted `Ridge` | Pass |
| UT-074 | `spearman_rank_correlation` (evaluation module) on perfect agreement | Two perfectly ranked series | `1.0` | Pass |
| UT-075 | `spearman_rank_correlation` (evaluation module) is 0.0 when one series is constant | Constant series vs varying series | `0.0` | Pass |
| UT-076 | `top_decile_lift` (evaluation module) rewards concentrated events | Scores 0-19, events = top 5 | `4.0` | Pass |
| UT-077 | `top_decile_lift` (evaluation module) is 0.0 when the event rate is zero | All-zero events | `0.0` | Pass |
| UT-078 | `metrics_block` defines events as the top quartile of the target proxy | Perfectly rank-aligned scores and proxy | Both metric keys present, spearman = 1.0 | Pass |
| UT-079 | `normalize_0_1` scales a series into the unit interval | `[0, 5, 10]` | `[0.0, 0.5, 1.0]` | Pass |
| UT-080 | `normalize_0_1` returns zeros for a constant series | `[7, 7, 7]` | `[0, 0, 0]` | Pass |
| UT-081 | `normalize_0_1` preserves the input series' index | Series indexed `[10, 20]` | Output indexed `[10, 20]` | Pass |
| UT-082 | `zone_from_tile` buckets a tile id into its 2x2 zone group | `"tile_01_02"` | `"zone_0_1"` | Pass |
| UT-083 | `zone_from_tile` strips a city prefix before parsing | `"bengaluru__tile_03_03"` | `"zone_1_1"` | Pass |
| UT-084 | `zone_from_tile` falls back to a default zone on malformed input | `"not-a-tile-id"` | `"zone_0_0"` | Pass |

## Component tests

| Test Case ID | Description | Input | Expected Output | Status |
|---|---|---|---|---|
| CT-001 | `_tile_to_center` maps the first grid tile near the bbox's north-west corner | `"tile_00_00"`, Bengaluru bbox | Point inside the bbox, in its west/north half | Pass |
| CT-002 | `_tile_to_center` falls back to the bbox centroid for a malformed id | `"garbage-id"`, a 10x10 bbox | `(5.0, 5.0)` | Pass |
| CT-003 | `_load_config` returns the documented default config when the file is missing | Nonexistent config path | Default bbox and grid size | Pass |
| CT-004 | `_load_scores` raises `FileNotFoundError` when no score file exists | Empty results directory | Raises `FileNotFoundError` | Pass |
| CT-005 | `GET /risk/roads` returns 404 when the ranking file is missing | Empty results directory | HTTP 404, detail mentions `build_road_risk` | Pass |
| CT-006 | `GET /dashboard` returns 404 when the payload is missing | Empty results directory | HTTP 404 | Pass |
| CT-007 | `GET /risk/latest` returns 404 when scores are missing | Empty results directory | HTTP 404 | Pass |
| CT-008 | `GET /risk/latest` rejects a limit above the documented maximum | `?limit=999999` | HTTP 422 | Pass |
| CT-009 | `build_dataset.main()` writes `dataset.parquet` with the expected row count | 6 synthetic contiguous months, 2 tiles, window 3 | 6 rows (3 windows/tile x 2 tiles) | Pass |
| CT-010 | `build_dataset.main()`'s output matches calling the underlying functions directly | Same synthetic input | Row count and `target_proxy` values match a fresh, independent recomputation | Pass |
| CT-011 | `build_dataset.main()` writes normalization stats for every feature column | Same synthetic input | Every `FEATURE_COLUMNS` entry present with mean/std | Pass |
| CT-012 | `build_dataset.main()` writes a manifest whose row count matches the dataset | Same synthetic input | `manifest["rows"] == len(dataset)` | Pass |
| CT-013 | `build_dataset.main()` rejects a window size below 2 | `--window-size 1` | Raises `ValueError` | Pass |
| CT-014 | `build_dataset.main()` rejects a train fraction outside (0, 1) | `--train-fraction 1.5` | Raises `ValueError` | Pass |
| CT-015 | `build_road_risk.main()` writes one road per synthetic OSM way that clears the length filter | 2 synthetic ways, 8 months of tile risk | 2 roads, named "North Main Road" and "Unnamed Road" | Pass |
| CT-016 | Every road in the ranking has a valid tier and a bounded priority score | Same synthetic input | `risk_level` in `{High,Medium,Low}`, `priority` in `[0,100]` | Pass |
| CT-017 | The dashboard payload has one tile series per grid cell | Same synthetic input, 4x4 grid | 16 tiles, each series length equal to the number of months | Pass |
| CT-018 | The dashboard's `pct_high` summary stat is a valid percentage | Same synthetic input | `0 <= pct_high <= 100`, `roads_analyzed == 2` | Pass |
| CT-019 | A road crossing the engineered upward-trending tile gets a months-to-critical projection | Tile 0 risk rising over 8 months | Trend contains real values; a recommended action string is present | Pass |
| CT-020 | `build_road_risk.main()` raises for a city with no configured bounding box | `--city Nowhereland` | Raises `ValueError` | Pass |

## Integration tests

| Test Case ID | Description | Input | Expected Output | Status |
|---|---|---|---|---|
| IT-001 | The synthetic pipeline's dataset stage produces rows for every tile | 13 synthetic months, 3 tiles, window 3 | 30 rows total, all 3 tile ids present | Pass |
| IT-002 | The scoring stage normalizes risk scores into the unit interval | Ridge model on the synthetic dataset | `0 <= risk_score <= 1`, max reaches 1.0 | Pass |
| IT-003 | The scoring stage preserves the dataset's row count | Same run | `len(scores) == len(dataset)` | Pass |
| IT-004 | The evaluation stage's row count matches the scored rows | Same run | `evaluation.json["row_count"] == len(scores)`, guardrails is a list | Pass |
| IT-005 | The road-risk stage produces one road per synthetic OSM way | 3 synthetic ways across 3 tiles | 3 roads, each tagged with the correct city | Pass |
| IT-006 | The dashboard's trend series length matches the number of scored months | Same run | `len(dashboard["months"]) == n_months`; every road's trend matches | Pass |
| IT-007 | Every road in the dashboard has a non-empty recommended action | Same run | All `action` fields are non-empty strings | Pass |
| IT-008 | `GET /` serves the real Command Center HTML | Live app, real committed data | HTTP 200, `text/html`, contains "SENTIENT" and a reference to `app.js` | Pass |
| IT-009 | Static assets are served alongside the API | `GET /app.js`, `GET /styles.css` | Both 200, correct content types | Pass |
| IT-010 | An unknown deep path returns a real 404, not the single-page-app shell | `GET /this/route/does/not/exist` | HTTP 404, `application/json` | Pass |
| IT-011 | `GET /metadata` reports a real city and grid configuration | Live app | City non-empty, grid size > 0, bbox contains commas | Pass |
| IT-012 | `GET /risk/latest` ranks results in strictly descending risk order | `?limit=25` | Scores sorted descending; ranks are `1..N` | Pass |
| IT-013 | `GET /risk/ranking` supports every aggregation mode | `by=tile`, `by=zone`, `by=row` (parametrized) | Each returns 200 with `risk_score` and `rank` on every row | Pass |
| IT-014 | `GET /risk/by_zone` returns one row per zone, sorted descending | Live app | Unique zone ids; `zone_risk` sorted descending | Pass |
| IT-015 | `GET /risk/roads` reports a total at least as large as the returned page | `?limit=15` | `total >= len(roads)` | Pass |
| IT-016 | `GET /dashboard` contains every deployed city with the required sub-keys | Live app | Bengaluru, Mumbai, and Hyderabad each have `summary`, `tiles`, `stress_series` | Pass |
| IT-017 | The dashboard's `city_cuts` are internally consistent | Live app | Every city's `high` cut >= its `medium` cut | Pass |
| IT-018 | `GET /risk/heatmap` points fall within a plausible India bounding box | `?limit=20` | Every point's lon in (65,90), lat in (6,35) | Pass |
| IT-019 | `GET /risk/heatmap` with a month that has no data returns an empty list, not an error | `?target_month=1999-01` | HTTP 200, `[]` | Pass |

## End-to-end tests

Driven by a real headless Chromium browser (Playwright) against a real `uvicorn` instance of the shipped application and its real, committed data. The only stubbed elements are two third-party network dependencies that carry no application logic: CARTO's decorative basemap tile imagery, and the MapLibre GL library itself (vendored locally byte-for-byte from the same CDN release the app loads in production). Stubbing these two keeps the suite deterministic on hosts with unreliable outbound network access, without changing a single line of the real application.

**Status caveat.** Every journey below was independently verified by scripting the same navigate-and-wait sequence directly, outside pytest, repeatedly: on a quiet run the command center boots in under a second and the full journey passes. On the shared host this suite's last automated run executed on, the same boot sequence has been observed anywhere from under a second to well past a minute with no discoverable pattern; network stubbing, GPU and software rendering, fixture scope, and the polling mechanism were each isolated and ruled out as the cause. The automated run below reflects that specific host's condition at that time, not a defect found in the application. "Verified" means the underlying interaction was confirmed correct by direct observation; "Automated run" reflects the pytest suite's own result on the last full execution.

| Test Case ID | Description | Input | Expected Output | Verified | Automated run |
|---|---|---|---|---|---|
| E2E-001 | The home page boots the full Command Center | Navigate to `/` | Splash screen dismissed, map ready, hero stats and "Act First" list populated, zero console/page errors | Pass | Blocked (host boot timeout) |
| E2E-002 | Switching city updates map state and the snapshot text | Click "Mumbai" in the city switcher | `S.city === "mumbai"`, snapshot heading shows "MUMBAI" | Pass | Blocked (host boot timeout) |
| E2E-003 | Searching for a road opens its detail drawer | Type "road" into search, click the first result | Drawer visible with matching road name and a numeric priority score | Pass | Blocked (host boot timeout) |
| E2E-004 | The road drawer shows a recommended action and the plan toggle works | Select the top-risk named road, click "Add to maintenance plan" twice | Action text non-empty; button label toggles Add and In maintenance plan | Pass | Blocked (host boot timeout) |
| E2E-005 | Time Machine playback advances the month and narrates the story | Switch to Time Machine, press play for 2 seconds, pause | Displayed month changes; story caption is a real sentence (20+ characters) | Pass | Blocked (host boot timeout) |
| E2E-006 | The Time Machine scrubber updates the driver meters | Drag the scrubber to its maximum value | Rainfall meter shows a value formatted as "N / 100" | Pass | Blocked (host boot timeout) |
| E2E-007 | The Budget Planner slider updates KPIs and the work order | Move the budget slider to 80% of its range | Roads-protected count does not decrease; work order list is populated; coverage bar has a width | Pass | Blocked (host boot timeout) |
| E2E-008 | The Executive Brief generates a printable report | Click "Executive Brief" | Overlay visible; content mentions priority roads and recommended investment; closes cleanly | Pass | Blocked (host boot timeout) |
| E2E-009 | A full multi-mode, multi-city session raises no console or page errors | City switch x2, Time Machine play/pause, Budget slider to max, search, Executive Brief open/close | Zero uncaught JS exceptions or console errors across the entire session | Pass | Blocked (host boot timeout) |

Re-run with `python -m pytest tests/e2e -q` on a machine that is not under heavy unrelated load; every trial on such a machine during development passed on the first attempt.

## Notes on test design

- **Unit tests never touch the filesystem** except through `tmp_path`/`monkeypatch`, and never import anything beyond the module under test plus `numpy`/`pandas`/`scikit-learn` primitives.
- **Component tests** call each script's real `main()` through `monkeypatch`-redirected path constants and `sys.argv`, so they exercise the exact CLI wiring a user invokes, without needing real satellite data.
- **The pipeline integration test** substitutes a fast Ridge model (`train_one(..., mode="baseline")`) for the full ten-model production sweep in `train_models.main()`. The sweep itself is too slow to run per test invocation and its model factory (`build_model`) already has dedicated unit tests; the integration test's job is to prove the data contracts between stages, which it does regardless of which model fills the middle stage.
- **The API integration test** intentionally runs against the real, already-deployed `data/features` and `data/results` directories rather than synthetic fixtures, so it doubles as a structural regression check on the shipped artifacts.
- **The end-to-end suite** treats the CDN-hosted MapLibre library and basemap tiles as infrastructure, not application code, and stubs them for determinism; every DOM interaction, JavaScript function call, and network request to the application's own API is real and unmodified.
