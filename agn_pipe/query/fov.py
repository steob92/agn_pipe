from astropy.coordinates import SkyCoord
from astropy.io import fits
from ..data.catalogs import read_3hwc, read_starcat
from astropy import units as u
from astropy.table import Table
from typing import Union, Optional


def query_3hwc(
    ra: float, dec: float, search_radius: Union[str, u.Quantity] = "1.5 deg"
) -> Table:
    """Query the 3HWC catalog for nearby sources

    Args:
        ra : Right Acession of the region of interst
        dec : Declinations of the region of interst
        search_radius : Query radius. Defaults to 1.5 degrees

    Returns:
        astropy.table.Table: Table of sources matching search criteria

    Examples:
        >>> query_3hwc(83.6287, 22.0147)
    <Table length=1>
        name         ra     dec        ts          flux     index    radius
        str14      float64 float64   float64      float64   float64  float64
    -------------- ------- ------- ------------ ----------- -------- -------
    3HWC J0534+220 83.6279 22.0243 35736.499681 2.34204e-13 -2.57949     0.0

    Raises:
        ValueError: If search_radius is not convertable to degrees
    """

    try:

        if isinstance(search_radius, str):
            search_radius = u.Quantity(search_radius).to("deg")
        else:
            search_radius = search_radius.to("deg")

    except ValueError as e:
        raise ValueError(
            f"Error converting search radius. Make sure unit is correct:\n{e}"
        )

    tab_3hwc = read_3hwc()
    tab_mask = (tab_3hwc["ra"] - ra) ** 2 + (
        tab_3hwc["dec"] - dec
    ) ** 2 < search_radius.value**2

    return tab_3hwc[tab_mask]


def query_starcat(
    ra: float,
    dec: float,
    search_radius: Union[str, u.Quantity] = "1.5 deg",
    magnitude: Optional[float] = 8.0,
) -> Table:
    """Query the 3HWC catalog for nearby sources

    Args:
        ra : Right Acession of the region of interst
        dec : Declinations of the region of interst
        search_radius : Query radius. Defaults to 1.5 degrees
        magnitude : Limiting magnitude. Defaults to 8.0

    Returns:
        astropy.table.Table: Table of sources matching search criteria

    Examples:
        >>> query_starcat(83.6287, 22.0147)
        <Table length=8>
            ra       dec       id     mag
        float64   float64  float64 float64
        --------- --------- ------- -------
        82.564146 22.540258 25779.0    7.77
        82.680615 22.462255 25806.0    6.29
        83.272304 23.142298 26015.0    7.91
        83.918861 21.403257 26272.0    7.56
        84.109928 21.993108 26328.0    6.88
        84.26036 20.730788 26381.0    7.68
        84.411191 21.142549 26451.0    2.97
        84.86295 21.762929 26616.0    6.42

    Raises:
        ValueError: If search_radius is not convertable to degrees

    """

    try:

        if isinstance(search_radius, str):
            search_radius = u.Quantity(search_radius).to("deg")
        else:
            search_radius = search_radius.to("deg")

    except ValueError as e:
        raise ValueError(
            f"Error converting search radius. Make sure unit is correct:\n{e}"
        )

    tab_starcat = read_starcat()
    tab_mask = (tab_starcat["ra"] - ra) ** 2 + (
        tab_starcat["dec"] - dec
    ) ** 2 < search_radius.value**2
    tab_mask &= tab_starcat["mag"] < magnitude

    return tab_starcat[tab_mask]


def get_exclusion_regions(
    ra: float,
    dec: float,
    search_radius: Union[str, u.Quantity] = "1.5 deg",
    magnitude: Optional[float] = 8.0,
    exclusion: Union[str, u.Quantity] = "0.35 deg",
) -> List[float, float, float]:
    """Get the exclusins regions of the FoV

    Args:
        ra : Right Acession of the region of interst
        dec : Declinations of the region of interst
        search_radius : Query radius. Defaults to 1.5 degrees
        magnitude : Limiting magnitude of stars to exclude. Defaults to 8.0
        exclusion : default exclusion radius. Defaults to 0.35 degrees


    Returns:
        List of [ra, dec, exclusion radius] of source to be excluded

    Examples:
        >>> get_exclusion_regions(83.6287, 22.0147)

    """
    pass
