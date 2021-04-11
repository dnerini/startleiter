import numpy as np
import xarray as xr
from startleiter import utils

def test_to_wind_components():
    n = 100
    speed = np.random.randint(0, 100, n)
    direction = np.random.randint(0, 360, n)
    level = np.arange(n)
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
