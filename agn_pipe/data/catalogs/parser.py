from astropy.io.misc import yaml
from astropy.table import Table
from astropy.io import fits


def parse_3hwc():
    """Parse the 3HWC yaml file

    File originally obtained from:
    https://data.hawc-observatory.org/datasets/3hwc-survey/3HWC.yaml

    Outputs to a fits file with 3HWC stored as an astropy.table.Table

    Args:
        None

    Returns:
        None
    """

    with open("./3HWC.yaml") as f:
        data = yaml.load(f)

        tab = {
            "name": [],
            "ra": [],
            "dec": [],
            "ts": [],
            "flux": [],
            "index": [],
            "radius": [],
        }

        for datum in data:

            # print (datum)
            tab["name"].append(datum["name"])
            tab["ra"].append(datum["RA"])
            tab["dec"].append(datum["Dec"])
            tab["ts"].append(datum["TS"])
            if "flux measurements" in datum:
                tab["flux"].append(datum["flux measurements"][0]["flux"])
                tab["index"].append(datum["flux measurements"][0]["index"])
                tab["radius"].append(datum["flux measurements"][0]["assumed radius"])
            else:
                tab["flux"].append(0)
                tab["index"].append(0)
                tab["radius"].append(0)

        tab = Table(tab)
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(tab)])
        hdul[1].name = "3HWC"
        hdul.writeto("./3HWC.fits", overwrite=True)


if __name__ == "__main__":
    parse_3hwc()
