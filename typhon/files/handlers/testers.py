__all__ = [
    'check_lat_lon',
]


def check_lat_lon(data):
    # check latitudes:
    if (data["lat"].min().item(0) < -90
            or data["lat"].max().item(0) > 90):
        values = data["lat"].min().item(0), data["lat"].max().item(0)
        raise ValueError(
            f"Latitudes are out of bounds (not in [-90, 90]): {values}"
        )

    if (data["lon"].min().item(0) < -180
            or data["lon"].max().item(0) > 180):
        values = data["lon"].min().item(0), data["lon"].max().item(0)
        raise ValueError(
            f"Longitudes are out of bounds (not in [-180, 180]): {values}"
        )