# Fuel type mapping for Kazakhstan

Scripts for building a 20 m fuel type map (Scott & Burgan FBFM40 codes) from
Sentinel-2, Sentinel-1, Copernicus DEM, ETH canopy height and ESRI land cover.
Everything except the clustering itself is pulled from Google Earth Engine.

The pipeline was first developed for the Almaty / Jetisu region and is now being
scaled up to the whole country. Both setups live in the same code, switched by
`STACK_MODE` in `config.py`.

## How it works

1. **Features.** For every 20 m pixel we compute 12 features:
   - NDVI, NDMI, SAVI, NBR from a June–August 2022 Sentinel-2 median
     (clouds masked with Cloud Score+)
   - elevation and slope from Copernicus GLO-30
   - Sentinel-1 VV and VH, median of May–September 2021–2022
   - season productivity for April–October 2022: mean NDVI, max NIRv, NDVI amplitude
   - canopy height (ETH 2020)

   ESRI LULC 2022 is carried along as a 13th band and is only used as a mask.
2. **Non-fuel mask.** Water, cropland, built-up, bare ground and snow (ESRI
   classes) are written straight to NB codes (91–99) and kept out of clustering.
3. **Clustering.** A random sample of fuel pixels is standardised, reduced with
   PCA (95% variance) and clustered with k-means. The fitted models are then
   applied to every pixel.
4. **Labelling.** Each cluster gets an FBFM40 code. In country mode this is done
   by rules on the cluster centroid (see below); for the Almaty run it is a hand-made
   table.

## Files

| file | what it does |
|---|---|
| `config.py` | paths, AOI, tile grid, band scales, model parameters |
| `acquire.py` | Earth Engine downloads (`stack` for the country, older per-layer commands for Almaty) |
| `features.py` | Almaty only: indices, slope, alignment of layers to the S2 grid |
| `io_utils.py` | windowed reading of the feature stack |
| `cluster.py` | `sample`, `train`, `rollout`, `fingerprint` |
| `label.py` | cluster → FBFM40 |
| `validate.py` | coverage report, comparison with the old national map and reference points |
| `oldcross.py` | crosstab of clusters against the old FBFM40 map |

## Requirements

Python 3.10 with `earthengine-api`, `geemap`, `rasterio`, `numpy`,
`scikit-learn`, `joblib`, `pandas`, and GDAL command-line tools (≥ 3.4, built
with ZSTD). The Earth Engine project is set in `config.GEE_PROJECT`.

On a headless server, authenticate once with:

```
earthengine authenticate --auth_mode=notebook
```

## Running it for the whole country

Set `STACK_MODE = True` in `config.py`.

Quick check on a ~5 km patch first:

```
python acquire.py probe              # default point near Almaty
python acquire.py probe 76.95 43.08  # any lon/lat
```

Download the stack. The country is cut into 2°×2° tiles (117 of them touch
Kazakhstan). Tiles that already exist are skipped, so a crashed run can simply be
started again. Running the west and east halves in parallel roughly halves the
wall time:

```
nohup python acquire.py stack 46 40 68 56 > stack_west.log 2>&1 &
nohup python acquire.py stack 68 40 88 56 > stack_east.log 2>&1 &
python acquire.py vrt                # rebuild data/stack_kz.vrt once both finish
```

A single tile takes about 30 minutes, so the whole country takes 2–3 days on one
process. On disk it comes to roughly 150 GB.

Then:

```
python validate.py coverage   # per-band coverage and value ranges
python cluster.py all         # sample -> train -> rollout -> fingerprint
python label.py rules         # print the rule decision for each cluster
python label.py               # write output/fueltype_labeled_kz_k40.tif
```

## Running the Almaty region (old layout)

With `STACK_MODE = False`:

```
python acquire.py all
python features.py indices
python features.py terrain
python features.py align
python cluster.py all
python label.py
```

## Stack format

All tiles share one grid: EPSG:4326, pixel size 2°/11132 ≈ 0.00017966°
(about 20 m), with tile corners on even degrees. Neighbouring tiles therefore line up
exactly and the VRT has no seams.

Bands are int16, stored as `round(value * scale)` with nodata −32768.

| band | scale |
|---|---|
| NDVI, NDMI, SAVI, NBR, INDVI, NIRv, NDVI_amp | 1000 |
| VV, VH (dB) | 100 |
| slope (deg) | 100 |
| elevation (m), canopy_h (m), lc | 1 |

Files are ZSTD-compressed (level 9, predictor 2), use 512×512 tiles and carry
internal overviews at 4/16/64. We tested int16 against float32 and uint8 on the
real data. Against float32, int16 is about 4× smaller on disk, and it moves
fewer than 0.1% of pixels to a different cluster.

## Labelling rules

`label.py` looks at four centroid values: canopy height, mean season NDVI
(INDVI), NDMI as a dry/humid switch, and elevation.

- canopy ≥ 6 m: timber. TL3 above 1400 m (spruce), otherwise TU1/TU2.
- canopy 3–6 m: shrub. SH1/SH2 when dry, SH3/SH8 when humid.
- canopy 1–3 m: grass-shrub, GS1–GS4.
- below that: grass, GR1–GR9 by INDVI. Wet lowland reed beds go to GR9.
- INDVI < 0.08: NB9.

The thresholds were tuned by eye against the Almaty clusters. They are a
starting point, not a calibrated model.

## Known issues

- **Canopy height in the mountains.** ETH overestimates canopy height on steep
  terrain. In the Almaty test, more than half of the alpine meadow pixels came out
  at 6 m or taller. Above 1000 m, `acquire.py` therefore keeps canopy height only
  where Hansen tree cover is ≥ 10% or WorldCover says tree or shrub.
- **Season.** One summer window is used for the whole country. That suits the
  north, but in the southern deserts green-up peaks in spring.
- **Sentinel-1.** 2022 alone has gaps because S1B failed in December 2021,
  which is why S1 uses two seasons.
- **Earth Engine quota.** The project has hit the non-commercial quota before.
  If downloads start failing or crawling, that is the first thing to check.
- **No accuracy assessment yet.** `reference_almaty_jetisu.csv` holds 2015
  land cover classes, not fuel models. A proper check needs reference points
  labelled with FBFM40 codes.
- `COPERNICUS/DEM/GLO30` is deprecated in favour of `GLO30_2024_1`. The old one
  still works; switching is on the list.

## Outputs

- `output/clusters_<aoi>_k<k>.tif`: cluster id per pixel, NB codes for
  non-fuel, 255 for nodata
- `output/fueltype_labeled_<aoi>_k<k>.tif`: FBFM40 codes
- `output/{scaler,pca,kmeans}_<aoi>.joblib`, `X_sample_<aoi>.npy`: fitted
  models and the training sample
