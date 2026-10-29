#!/usr/bin/env bash
# AZ parity ship (zephyr): statewide 100 m mosaics, 10 m VRT, zip (store), md5, push to the GCS outbox and,
# if an rclone Drive remote exists on this node, to the Drive folder. Same packaging as nm_parity_ship.sh
# (zoran, 2026-10-19) and the NV delivery (notes/NV_SHIP_DRIVE_HANDOFF_2026-10-08.md). No rendering, no
# smoothing, no training. Run only after the user's ship decision. Requires $OUT/README.txt (write it from
# the template in notes/ZEPHYR_AZ_COMPLETION_HANDOFF.md with the eval numbers filled in) and gdal CLI on PATH
# (module load gdal, or export the user-space GDAL env first). The 100 m mosaics are rebuilt only if the difference
# raster is missing, so a prior inspection build (handoff step 7) is reused. Never overwrites an existing zip.
set -euo pipefail
H=/data/ssd2/handily; HUC=$H/huc8; HC=/project/handily/code/handily-nv
ARM0=gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod
ARM=gnn_conus_monitoring_water_src_r1e_m20_ord_adwrlbl_parity
N=$H/az/regional/wells/az_huc8_84.txt
Q=$H/share/ddf_qgis/parity_az; OUT=$H/share/drive_outbox/az; MOS=$HUC/_mosaics
GCS_OUT=gs://wudr/handily/azadwr/drive_outbox; GCS_Q=gs://wudr/handily/azadwr/share/ddf_qgis/parity_az
FID=1Z9XBJV0KWkymFAeQjJYBygu9nkZgqDQe
mkdir -p $Q $OUT $MOS
command -v gdalbuildvrt > /dev/null || { echo "gdal CLI not on PATH (module load gdal?)"; exit 1; }
[ -f $OUT/README.txt ] || { echo "write $OUT/README.txt first (template in the handoff)"; exit 1; }
echo "SHIP START $(date) arm $ARM; df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
# 1. 100 m statewide mosaics: parity DTW + sigma over 84 basins; shipped w25_prod DTW over its 19 AZ basins on the
#    parity grid, and shipped - parity on that 19-basin footprint only
if [ ! -f $Q/az_dtw_shipped_minus_parity_19basins.tif ]; then
: > $Q/parity_dtw_list.txt; : > $Q/parity_sigma_list.txt; : > $Q/shipped_dtw_list.txt
for b in $(cat $N); do
  echo $HUC/$b/gnn/$ARM/gnn_dtw_100m.tif >> $Q/parity_dtw_list.txt
  echo $HUC/$b/gnn/$ARM/gnn_sigma_100m.tif >> $Q/parity_sigma_list.txt
  [ -f $HUC/$b/gnn/$ARM0/gnn_dtw_100m.tif ] && echo $HUC/$b/gnn/$ARM0/gnn_dtw_100m.tif >> $Q/shipped_dtw_list.txt
done
for f in $(cat $Q/parity_dtw_list.txt $Q/parity_sigma_list.txt $Q/shipped_dtw_list.txt); do [ -f $f ] || { echo "MISSING $f"; exit 1; }; done
echo "lists: $(wc -l < $Q/parity_dtw_list.txt) parity dtw, $(wc -l < $Q/parity_sigma_list.txt) sigma, $(wc -l < $Q/shipped_dtw_list.txt) shipped (expect 84/84/19)"
gdalbuildvrt -q -overwrite -input_file_list $Q/parity_dtw_list.txt $Q/az_dtw_parity_statewide_100m.vrt
gdalbuildvrt -q -overwrite -input_file_list $Q/parity_sigma_list.txt $Q/az_sigma_parity_statewide_100m.vrt
gdalbuildvrt -q -overwrite -input_file_list $Q/shipped_dtw_list.txt $Q/az_dtw_shipped_19basins_100m.vrt
TE=$(uv run --directory $HC python -c "import rasterio;b=rasterio.open('$Q/az_dtw_shipped_19basins_100m.vrt').bounds;print(b.left,b.bottom,b.right,b.top)")
gdalbuildvrt -q -overwrite -te $TE -input_file_list $Q/parity_dtw_list.txt $Q/az_dtw_parity_19basins_100m.vrt
for n in az_dtw_parity_statewide_100m az_sigma_parity_statewide_100m az_dtw_shipped_19basins_100m az_dtw_parity_19basins_100m; do
  gdal_translate -q -of COG -co COMPRESS=DEFLATE -co PREDICTOR=3 -co BIGTIFF=IF_SAFER $Q/$n.vrt $Q/$n.tif
done
gdal_calc.py -A $Q/az_dtw_shipped_19basins_100m.tif -B $Q/az_dtw_parity_19basins_100m.tif --outfile=$Q/az_dtw_shipped_minus_parity_19basins.tif --calc="A-B" --NoDataValue=-9999 --co COMPRESS=DEFLATE --co TILED=YES --quiet --overwrite
gdalinfo -stats $Q/az_sigma_parity_statewide_100m.tif > /dev/null
gdalinfo -stats $Q/az_dtw_parity_statewide_100m.tif | grep -E "Size is|STATISTICS_(MINIMUM|MAXIMUM|MEAN)" | tr '\n' ' '; echo
echo "100 m mosaics done $(date)"
else echo "100 m mosaics already built"; fi
# 2. 10 m tile list + VRT (zip-relative paths)
for b in $(cat $N); do ls $HUC/$b/gnn/$ARM/*; done | grep -v -E "presmooth|\.ovr$" | sed "s#^$H/##" > $OUT/az_dtw_parity_10m_tiles.lst
echo "tile list: $(wc -l < $OUT/az_dtw_parity_10m_tiles.lst) files (expect 84 x 16 = 1344)"
grep gnn_dtw_10m.tif $OUT/az_dtw_parity_10m_tiles.lst > $OUT/dtw10_tiles.txt
sed "s#^#$H/#" $OUT/dtw10_tiles.txt > $OUT/dtw10_tiles_abs.txt
gdalbuildvrt -q -overwrite -input_file_list $OUT/dtw10_tiles_abs.txt $MOS/az_dtw_parity_10m.vrt
[ "$(grep -c SourceFilename $MOS/az_dtw_parity_10m.vrt)" -eq 84 ] || { echo "10 m VRT has $(grep -c SourceFilename $MOS/az_dtw_parity_10m.vrt) sources, expected 84"; exit 1; }
sed "s#relativeToVRT=\"0\">$H/#relativeToVRT=\"1\">#" $MOS/az_dtw_parity_10m.vrt > $OUT/az_dtw_parity_10m.vrt
echo "10 m VRT: $(grep -c 'relativeToVRT="1"' $OUT/az_dtw_parity_10m.vrt) relative refs"
# 3. zips (store; every raster is already a deflate COG)
for z in az_dtw_parity_10m_tiles.zip az_dtw_parity_statewide_100m.zip; do [ -f $OUT/$z ] && { echo "$OUT/$z exists -- refusing to overwrite"; exit 1; }; done
cd $H && zip -0 -q $OUT/az_dtw_parity_10m_tiles.zip -@ < $OUT/az_dtw_parity_10m_tiles.lst
cd $OUT && zip -0 -q az_dtw_parity_10m_tiles.zip az_dtw_parity_10m.vrt README.txt
cp $OUT/README.txt $Q/README.txt && cd $Q && zip -0 -q $OUT/az_dtw_parity_statewide_100m.zip az_dtw_parity_statewide_100m.tif az_dtw_shipped_minus_parity_19basins.tif az_sigma_parity_statewide_100m.tif az_sigma_parity_statewide_100m.tif.aux.xml README.txt
echo "zips done $(date): tiles zip entries $(unzip -l $OUT/az_dtw_parity_10m_tiles.zip | tail -1), 100 m zip entries $(unzip -l $OUT/az_dtw_parity_statewide_100m.zip | tail -1)"
cd $OUT && md5sum az_dtw_parity_10m_tiles.zip az_dtw_parity_statewide_100m.zip > SHA_az.md5 && cat SHA_az.md5; ls -l $OUT/*.zip
# 4. push: QGIS mosaics + outbox to GCS (free from zephyr); Drive only if an rclone remote named gdrive exists here
gsutil -m cp $Q/az_*_100m.tif $Q/az_dtw_shipped_minus_parity_19basins.tif $Q/README.txt $GCS_Q/
gsutil -m cp $OUT/az_dtw_parity_statewide_100m.zip $OUT/SHA_az.md5 $OUT/az_dtw_parity_10m_tiles.zip $OUT/README.txt $GCS_OUT/
gsutil ls -l $GCS_OUT/
if command -v rclone > /dev/null && rclone listremotes | grep -q '^gdrive:'; then
  for f in az_dtw_parity_statewide_100m.zip SHA_az.md5 az_dtw_parity_10m_tiles.zip; do echo "== upload $f $(date)"; rclone copy --drive-root-folder-id $FID --drive-chunk-size 64M --stats 60s --stats-one-line $OUT/$f gdrive:; done
  echo "== lsl $(date)"; rclone lsl --drive-root-folder-id $FID gdrive:
else echo "no rclone gdrive remote on this node: zips are in $GCS_OUT; the user moves them to Drive"; fi
echo "SHIP DONE $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
