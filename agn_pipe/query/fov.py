from astropy.coordinates import SkyCoord
from astropy.io import fits
from ..data.catalogs import read_3hwc, read_starcat, read_vtscat
from astropy import units as u
from astropy.table import Table
from typing import Union, Optional


def query_vtscat(
    ra: float, dec: float, search_radius: Union[str, u.Quantity] = "1.5 deg"
) -> Table:
    """Query the VTSCat catalog for nearby sources

    Args:
        ra : Right Acession of the region of interst
        dec : Declinations of the region of interst
        search_radius : Query radius. Defaults to 1.5 degrees

    Returns:
        astropy.table.Table: Table of sources matching search criteria

    Examples:
        >>> query_vtscat(83.6287, 22.0147)
        <Table length=2>
            name         ra         dec
        str27      float64     float64
        ----------- ----------- ------------
        Crab nebula   83.633083      22.0145
        Crab pulsar 83.63307625 22.014493278


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

    tab_vts = read_vtscat()
    tab_mask = (tab_vts["ra"] - ra) ** 2 + (
        tab_vts["dec"] - dec
    ) ** 2 < search_radius.value**2

    return tab_vts[tab_mask]


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
) -> list[float, float, float]:
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
        [[83.6279, 22.0243, '0.35 deg'],
        [83.633083, 22.0145, '0.35 deg'],
        [83.63307625, 22.014493278, '0.35 deg'],
        [82.564146, 22.540258, '0.35 deg'],
        [82.680615, 22.462255, '0.35 deg'],
        [83.272304, 23.142298, '0.35 deg'],
        [83.918861, 21.403257, '0.35 deg'],
        [84.109928, 21.993108, '0.35 deg'],
        [84.26036, 20.730788, '0.35 deg'],
        [84.411191, 21.142549, '0.35 deg'],
        [84.86295, 21.762929, '0.35 deg']]

    """

    tab_3hwc = query_3hwc(ra, dec, search_radius)
    tab_vts = query_vtscat(ra, dec, search_radius)
    tab_starcat = query_starcat(ra, dec, search_radius, magnitude)

    exclusion_regions = []

    for entry in tab_3hwc:
        # Use the default if the radius is 0 otherwise use the catalog value
        radius = exclusion if entry["radius"] == 0 else entry["radius"]
        exclusion_regions.append([entry["ra"], entry["dec"], radius])
    # todo, vtscat extentions
    for entry in tab_vts:
        # Use the default if the radius is 0 otherwise use the catalog value
        # radius = exclusion if entry["radius"] == 0 else entry["radius"]
        radius = exclusion
        exclusion_regions.append([entry["ra"], entry["dec"], radius])

    for entry in tab_starcat:
        exclusion_regions.append([entry["ra"], entry["dec"], exclusion])

    return exclusion_regions
