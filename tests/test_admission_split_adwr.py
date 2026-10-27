"""ADWR registry attach for the admission split (use recovery + basin)."""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from build_admission_split import attach_adwr_registry  # noqa: E402


def _gwx():
    return gpd.GeoDataFrame(
        {
            "source_well_id": ["000001", "000002", "000003", "000004"],
            "well_use": ["domestic", None, None, None],
            "h_accuracy_class": ["plss_64"] * 4,
        },
        geometry=[Point(0, 0)] * 4,
        crs=4326,
    )


def _registry(tmp_path, ids=("000001", "000002", "000003", "000004")):
    reg = pd.DataFrame(
        {
            "REGISTRY_ID": list(ids),
            "WATER_USE": ["DOMESTIC", "MONITORING", "TEST, MONITORING", None],
            "BASIN_NAME": ["PHOENIX AMA", "VERDE", None, "YUMA"],
        }
    )
    p = tmp_path / "registry.parquet"
    reg.to_parquet(p)
    return str(p)


def test_monitoring_recovered_only_where_well_use_empty_and_leading(tmp_path):
    g = attach_adwr_registry(_gwx(), _registry(tmp_path))
    assert g["use_"].tolist() == ["domestic", "monitoring", "None", "None"]
    assert g["pod_basin"].tolist() == ["PHOENIX AMA", "VERDE", "", "YUMA"]
    assert (g["grnd_wtr_s"] == "").all()
    assert g["h_accuracy_class"].tolist() == ["plss_64"] * 4
    assert "water_use_raw" not in g.columns and "_merge" not in g.columns


def test_unjoined_rows_abort(tmp_path):
    with pytest.raises(SystemExit, match="did not join the ADWR registry"):
        attach_adwr_registry(
            _gwx(), _registry(tmp_path, ids=("000001", "000002", "000003", "000009"))
        )
