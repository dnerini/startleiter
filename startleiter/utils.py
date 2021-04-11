import metpy.calc as mpcalc
from metpy.units import units


def to_wind_components(dataset, inverse=False):
    """
    Convert wind direction and speed to (and from) wind components (u and v).

    Parameters
    ----------
    dataset: xarray.Dataset
    inverse: bool, optional

    Returns
    -------
    xarray.Dataset

    """
    dataset = dataset.copy()
    if not inverse:
        wind_components = mpcalc.wind_components(
            dataset["SKNT"].values * units.knots,
            dataset["DRCT"].values * units.deg
        )
        dataset["U"] = (dataset.coords, wind_components[0].magnitude)
        dataset["V"] = (dataset.coords, wind_components[1].magnitude)
        dataset["U"].attrs["units"] = "knot"
        dataset["U"] = dataset["U"].astype("float32")
        dataset["V"].attrs["units"] = "knot"
        dataset["V"] = dataset["V"].astype("float32")
        dataset = dataset.drop_vars(("SKNT", "DRCT"))

    else:
        dataset["SKNT"] = (dataset.coords, mpcalc.wind_speed(
            dataset["U"].values * units.knots,
            dataset["V"].values * units.knots
        ).magnitude)
        dataset["DRCT"] = (dataset.coords, mpcalc.wind_direction(
            dataset["U"].values * units.knots,
            dataset["V"].values * units.knots
        ).magnitude)
        dataset["SKNT"].attrs["units"] = "knot"
        dataset["SKNT"] = dataset["SKNT"].astype("float32")
        dataset["DRCT"].attrs["units"] = "deg"
        dataset["DRCT"] = dataset["DRCT"].astype("float32")
        dataset = dataset.drop_vars(("U", "V"))

    return dataset