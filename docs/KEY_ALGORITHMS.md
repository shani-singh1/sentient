# Key algorithms and logic

Short reference for the core logic in the pipeline and product. Written to be pasted into another document as is.

## 1. Monthly tile stress aggregation

Turns raw rasters into one row of stress metrics per tile per month.

```
for each city, year, month with overlapping raw data:
    load Sentinel-1, Sentinel-2, Landsat, nightlights, population rasters
    load ERA5 precipitation and temperature

    for each tile (i, j) in the 4x4 grid:
        crop each raster to the tile's pixel window
        compute NDVI = (NIR - Red) / (NIR + Red)
        compute NDWI = (Green - NIR) / (Green + NIR)
        convert Landsat thermal band to Kelvin if it looks like a raw digital number
        flood_fraction = fraction of Sentinel-1 pixels below -17 dB backscatter
        heat_exposure_fraction = fraction of thermal pixels above 305 K
        record mean, p90, and fraction-positive for every band of interest

    write one parquet file per month, one row per tile
```

## 2. Temporal window and target construction

Turns monthly tile rows into supervised learning rows. One row predicts one tile's next month from its previous `window_size` months.

```
for each tile, sorted by month:
    for each position i from window_size-1 to len-2:
        window = the window_size months ending at i
        next_row = the month right after the window
        if the window and next_row are not exactly consecutive calendar months:
            skip (do not bridge a data gap)

        target_proxy = precipitation_sum(next)
                     + 2.0 * heat_exposure_fraction(next)
                     + 3.0 * flood_fraction(next)

        for lag in 0..window_size-1:
            copy every raw feature at that lag into <feature>_lag<lag>

        for each fast-moving feature:
            delta1 = newest - previous month
            trend = (newest - oldest) / (window_size - 1)
            winmax, winstd = max and std across the window

        add sin/cos of the target month (seasonality)
        add cross terms: rain*flood, heat*temperature, rain*moisture, load*flood
        add stress_accum_* = 3-month rolling rain sum, flood sum, heat mean
        one-hot encode the city
```

Every feature is z-scored using mean/std fit only on rows at or before a time cutoff, then applied to the whole dataset, so validation-era statistics never leak into training.

## 3. Model selection

```
split rows by target_month, most recent val_fraction becomes validation (strict time split, no shuffling)

for each of 10 candidate models (Ridge baseline, HistGB, RandomForest,
                                  ExtraTrees x2, LightGBM x2, XGBoost,
                                  equal-weight blend, temporal-holdout stack):
    fit on train rows
    predict validation rows
    score: MAE, RMSE, R2, Spearman rank correlation, top-decile lift

pick the model with the best (top_decile_lift, then spearman, then R2)
save it as best_model.joblib
```

Top-decile lift asks: among the rows the model ranks in its riskiest 10%, how much more likely are they to actually land in the worst 25% of outcomes, compared to picking randomly. A value of 4.0 is the maximum this dataset allows.

The temporal-holdout stack avoids leakage in a stacked ensemble: base models train on the earliest slice of the training data, the meta-learner trains on their predictions over a later slice they never saw, then every base model is refit on the full training set for actual inference.

## 4. Risk scoring

```
predict on every row of the full dataset with the chosen model
risk_score = (prediction - min) / (max - min), clipped to [0, 1]
tile_risk = mean risk_score per tile
zone_risk = mean risk_score per 2x2 block of tiles
```

## 5. Tile risk to road risk

Bridges the modeled grid to something a person can act on.

```
tile_risk_map = mean risk_score per tile, using only the most recent 6 scored months

for each OSM way (road):
    walk its nodes in order, summing haversine distance between consecutive points
    map every node to a tile using its lon/lat and the city's bounding box
    road.risk_score = mean risk_score of every tile the road touches
    skip roads shorter than the minimum length

for each city:
    high_cut = 85th percentile of that city's road risk scores
    medium_cut = 50th percentile
    a road is High if risk_score >= high_cut, Medium if >= medium_cut, else Low

priority_score = percentile rank of a road's risk_score against every analyzed
                  road in its city, scaled to 0-100

select a display sample per city: about 40% High, 35% Medium, 25% Low,
    preferring named roads over unnamed ones at equal risk
```

## 6. Months to critical

Projects when a road's risk trend will cross into the critical band.

```
take the last 8 observed months of a road's risk trend
if fewer than 4 real observations exist: return unknown
if the latest value is already at or above the high cut: return 0
fit a straight line through the 8 points, take its slope
if the slope is flat or falling: return unknown (no rising trajectory)
months = ceil((high_cut - latest_value) / slope)
if months > 24: return unknown (too far out to be a useful projection)
return months
```

## 7. Stress driver attribution

Explains why a road is risky in plain language.

```
for each tile, take its most recent month's values for:
    rainfall, standing water, heat, surface moisture, corridor load, vegetation loss trend
rank each of these six signals as a percentile within its own city (0-100)

for each road:
    average the percentile scores of every tile it touches
    take the top 3 signals with a nonzero score
    the single top signal picks the recommended action from a lookup table
    (for example: High risk + standing water -> "Drainage audit within 2 weeks")
```

## 8. Time Machine playback (frontend)

Turns the monthly tile series into an animated, narrated replay.

```
on each month step:
    for every road: read its 3-month rolling average risk at this month
    classify each road as stable, watch, or critical against a threshold
    computed from that city's full 2020-2024 history (not a moving target),
    so bands do not flicker month to month
    recolor the road on the map to match its band

    for every tile: same 3-month smoothing, compare against that tile's own
    85th percentile across its full history, count how many tiles are
    "exceptional" this month

    read the three city-wide driver indices (rain, flood, heat) for this month
    pick the dominant driver and the phase (monsoon, pre-monsoon, dry) to
    generate one sentence describing what is happening and where
```

## 9. Budget planner (frontend)

Turns a budget number into a prioritized work order.

```
pool = every road that is not Low risk, sorted High-first then by risk score,
       named roads breaking ties over unnamed ones

spent = 0
picks = []
for each road in pool:
    cost = (length_km) * cost_per_km
    if spent + cost > budget: skip
    spent += cost
    picks.append(road)

roads_protected = len(picks)
km_resurfaced = sum of picked road lengths
reactive_cost_avoided = km_resurfaced * cost_per_km * (reactive_multiplier - 1) * failure_likelihood
coverage = (High-risk roads picked) / (all High-risk roads)
```
