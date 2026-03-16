# AOI

AOI tiling is the scaling utility in the repo. It converts a field layer into a set of bounded tiles
that can be processed independently.

## Current Supported CLI Path

```bash
handily aoi \
  --fields ~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp \
  --out-shp ~/data/IrrigationGIS/handily/aois/beaverhead_tiles.shp \
  --max-km2 625 \
  --buffer-m 1000 \
  --bounds -112.418 45.445 -112.353 45.49
```

## Method

- read fields, optionally clipped by bounds
- convert to an equal-area CRS
- build centroid buffers
- dissolve overlapping buffers
- recursively split the dissolved envelope into tiles not exceeding `max_km2`
- write the result as a shapefile with `aoi_id`

## Output

- AOI shapefile at `--out-shp`

## Notes

- The main CLI exposes centroid-buffer AOIs only.
- Additional helpers exist in `handily.aoi_split`, but they are not currently documented as supported
  operational interfaces.
