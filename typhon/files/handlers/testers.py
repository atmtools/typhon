__all__ = [
    'check_lat_lon',
]


def check_lat_lon(data):
    # check latitudes:
    if (data["lat"].min().item(0) < -90
            or data["lat"].max().item(0) > 90):
        raise ValueError(
            "Latitudes are out of bounds (not in [-90, 90])!"
        )

    if (data["lon"].min().item(0) < -180
            or data["lon"].max().item(0) > 180):
        raise ValueError(
            "Longitudes are out of bounds (not in [-180, 180])!"
        )