import numpy as np
import xarray as xr
from startleiter import utils

def test_to_wind_components():
    direction = np.array([90, 180, 270, 360])
    speed = np.ones(direction.shape) * 10
    level = np.arange(direction.size)
    ds1 = xr.Dataset(
        {
            "SKNT": ("level", speed),
            "DRCT": ("level", direction)
        },
        {
            "level": level,
        }
    )
    uv = utils.to_wind_components(ds1)
    ds2 = utils.to_wind_components(uv, True)
    xr.testing.assert_allclose(ds1, ds2)