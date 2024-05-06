from gammapy.data import DataStore
from astropy.time import Time
from typing import Union
from astropy import units as u


def query_datastore(
    store: str,
    ra: float,
    dec: float,
    search_cone: Union[str, u.Quantity, float] = 1.5,
    tstart: Union[Time, str, None] = None,
    tstop: Union[Time, str, None] = None,
) -> list[int]:
    """Query Data Store to get observations matching search criteria

    Query a data store to find observations within a distance from a specified RA and Dec.
    Option search in time.

    Args:
        ra  : Right Accesnion of the query location
        dec : Declination of the query location
        search_cone : Radius of the search region. Defaults to 1.5 degrees.
                      If a float is specified then the units is assumed to be degrees
        tstart  : Start time of the search query. Defaults to None, no start time
        tstop   : Stop time of the search query. Defaults to None, no stop time

    Returns:
        list of observation ids

    Raises:
        ValueError: If search_radius is not convertable to degrees
    """

    if isinstance(search_cone, str):
        try:
            search_cone = u.Quantity(search_cone)
        except ValueError as e:
            raise ValueError(f"Error converting search_cone to degrees:\n\t{e}")

    elif isinstance(search_cone, float):
        search_cone *= u.deg

    search_cone = search_cone.to("deg")

    data = DataStore.from_dir(store).obs_table
    # To do: fix the _PNT option
    # data = data[(data["RA_PNT"] - ra)**2 + (data["DEC_PNT"] - dec)**2 < search_cone.value**2]
    data = data[
        (data["RA_OBJ"] - ra) ** 2 + (data["DEC_OBJ"] - dec) ** 2
        < search_cone.value**2
    ]

    # Todo, some checks on times
    if tstart is not None:
        data = data[data["DATE-OBS"] > tstart]
    if tstop is not None:
        data = data[data["DATE-END"] < tstop]

    return list(data["OBS_ID"])
