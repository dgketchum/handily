import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import LineString

from handily.rem_sag import build_stream_evidence_support, relax_rem_downward


def _grid(values: np.ndarray) -> xr.DataArray:
    ny, nx = values.shape
    x = np.arange(0.5, nx + 0.5, 1.0)
    y = np.arange(ny - 0.5, -0.5, -1.0)
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def test_build_stream_evidence_support_intersects_stream_and_mask():
    mask = _grid(np.zeros((5, 5), dtype=np.uint8))
    vals = mask.values.copy()
    vals[2, 1:4] = 1
    mask = _grid(vals)
    streams = gpd.GeoDataFrame(
        {"geometry": [LineString([(1.5, 2.5), (3.5, 2.5)])]},
        geometry="geometry",
        crs="EPSG:5070",
    )
    support = build_stream_evidence_support(streams, mask)
    assert int(support.values.sum()) >= 2
    assert support.values[2, 2] == 1


def test_relax_rem_downward_pins_support_and_never_lifts():
    prior = _grid(
        np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 4, 4, 4, 5],
                [5, 4, 6, 4, 5],
                [5, 4, 4, 4, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )
    )
    support = _grid(np.zeros((5, 5), dtype=np.uint8))
    support.values[2, 2] = 1
    relaxed, info = relax_rem_downward(prior, support, return_info=True, max_iter=500, tol=1e-5)
    assert relaxed.values[2, 2] == 0.0
    assert np.nanmax(relaxed.values - prior.values) <= 1e-12
    assert np.nanmin(relaxed.values) >= 0.0
    assert relaxed.values[2, 1] < prior.values[2, 1]
    assert info.n_support == 1


def test_relax_rem_downward_keeps_boundary_when_requested():
    prior = _grid(np.full((6, 6), 10.0, dtype=np.float64))
    support = _grid(np.zeros((6, 6), dtype=np.uint8))
    support.values[3, 3] = 1
    relaxed = relax_rem_downward(prior, support, fix_boundary_to_prior=True, max_iter=300, tol=1e-5)
    assert np.allclose(relaxed.values[0, :], 10.0)
    assert np.allclose(relaxed.values[-1, :], 10.0)
    assert np.allclose(relaxed.values[:, 0], 10.0)
    assert np.allclose(relaxed.values[:, -1], 10.0)


def test_fac_hint_allows_more_sag():
    prior = _grid(np.full((7, 7), 8.0, dtype=np.float64))
    support = _grid(np.zeros((7, 7), dtype=np.uint8))
    support.values[3, 3] = 1
    hint = _grid(np.zeros((7, 7), dtype=np.float64))
    hint.values[3, 1:6] = 1.0
    no_hint = relax_rem_downward(prior, support, max_iter=300, tol=1e-5)
    with_hint = relax_rem_downward(
        prior,
        support,
        fac_hint_da=hint,
        fac_hint_scale=8.0,
        max_iter=300,
        tol=1e-5,
    )
    assert with_hint.values[3, 1] <= no_hint.values[3, 1] + 1e-12
    assert with_hint.values[3, 5] <= no_hint.values[3, 5] + 1e-12
