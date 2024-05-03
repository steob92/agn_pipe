from astropy.io import fits
from astropy.table import Table
from os import path
import pathlib
from numpy import loadtxt as np_loadtxt


def read_3hwc() -> Table:
    """Read 3HWC catalog from pregenerated fits file

    Args:
        None

    Returns:
        None
    """

    fname = path.join(pathlib.Path(__file__).parent.resolve(), "3HWC.fits")
    print(fname)
    with fits.open(fname) as hdul:
        tab = Table.read(hdul)
    return tab


def read_starcat() -> Table:
    """Read Hipparcos catalog

    Args:
        None

    Returns:
        None
    """
    fname = path.join(
        pathlib.Path(__file__).parent.resolve(), "Hipparcos_MAG8_1997.dat"
    )
    star_data = np_loadtxt(fname, usecols=(0, 1, 2, 3), skiprows=62)

    star_cat = Table(
        {
            "ra": star_data[:, 0],
            "dec": star_data[:, 1],
            "id": star_data[:, 2],
            "mag": star_data[:, 3],
        }
    )
    return star_cat
