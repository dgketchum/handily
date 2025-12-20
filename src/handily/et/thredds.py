from datetime import datetime
from urllib.parse import urlunparse

from pandas import DataFrame, date_range
from xarray import open_dataset


class GridMet:
    def __init__(self, variable=None, date=None, start=None, end=None, lat=None, lon=None):
        self.variable = variable
        self.date = date
        self.start = start
        self.end = end
        self.lat = lat
        self.lon = lon

        if isinstance(self.start, str):
            self.start = datetime.strptime(self.start, "%Y-%m-%d")
        if isinstance(self.end, str):
            self.end = datetime.strptime(self.end, "%Y-%m-%d")
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, "%Y-%m-%d")

        if self.date is not None:
            self.start = self.date
            self.end = self.date

        self.service = "thredds.northwestknowledge.net:8080"
        self.scheme = "http"

        self.kwords = {
            "etr": "daily_mean_reference_evapotranspiration_alfalfa",
            "pet": "daily_mean_reference_evapotranspiration_grass",
            "pr": "precipitation_amount",
            "srad": "daily_mean_shortwave_radiation_at_surface",
            "tmmn": "daily_minimum_temperature",
            "tmmx": "daily_maximum_temperature",
            "vs": "daily_mean_wind_speed",
            "sph": "daily_mean_specific_humidity",
        }

    def _date_index(self):
        idx = date_range(self.start, self.end, freq="d")
        return idx

    def _build_url(self):
        if self.variable == "elev":
            url = urlunparse(
                [
                    self.scheme,
                    self.service,
                    f"/thredds/dodsC/MET/{self.variable}/metdata_elevationdata.nc",
                    "",
                    "",
                    "",
                ]
            )
        else:
            url = urlunparse(
                [
                    self.scheme,
                    self.service,
                    f"/thredds/dodsC/agg_met_{self.variable}_1979_CurrentYear_CONUS.nc",
                    "",
                    "",
                    "",
                ]
            )
        return url

    def get_point_timeseries(self):
        url = self._build_url()
        url = url + "#fillmismatch"
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method="nearest")
        subset = subset.loc[dict(day=slice(self.start, self.end))]
        subset = subset.rename({"day": "time"})
        subset["time"] = self._date_index()
        time = subset["time"].values
        series = subset[self.kwords[self.variable]].values
        df = DataFrame(data=series, index=time)
        df.columns = [self.variable]
        return df

    def get_point_elevation(self):
        url = self._build_url()
        url = url + "#fillmismatch"
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method="nearest")
        elev = subset.get("elevation").values[0]
        return elev


# ========================= EOF =======================================================================================
