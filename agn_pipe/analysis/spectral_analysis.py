from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator, FluxPoints

# from gammapy.estimators.utils import resample_energy_edges
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    LogParabolaSpectralModel,
    EBLAbsorptionNormSpectralModel,
    SkyModel,
    create_crab_spectral_model,
)

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from astropy.io import fits


import uuid
from pathlib import Path
from ..query import get_exclusion_regions, query_datastore
from astropy.time import Time
from astropy.io import fits
from typing import Union, Optional, List
from math import ceil


class SpectralAnalysis:
    def __init__(
        self,
        datastore_name: str,
        ra: float,
        dec: float,
        source_name: str,
        tstart: Union[Time, str, None] = None,
        tstop: Union[Time, str, None] = None,
        base_path: Optional[str] = "./analysis/",
        scratch_path: Optional[str] = None,
        point_like: bool = True,
        obs_ids: Optional[List[int]] = None,
    ):
        """
        Initializes the SpectralAnalysis object.

        Parameters:
        -----------
            datastore_name : The name of the datastore.
            ra : The right ascension of the source in degrees.
            dec : The declination of the source in degrees.
            source_name : The name of the source.
            tstart : The start time for the analysis. Can be an astropy Time object, a string, or None.
            tstop : The stop time for the analysis. Can be an astropy Time object, a string, or None.
            base_path : The base path for the analysis. Defaults to "./analysis".
            scratch_path : The scratch path for the analysis. If None, a unique path will be generated.
            point_like : If the source is point-like. Defaults to True.
            obs_ids : The observation ids to use. If None, the observations will be queried.

        Returns:
        --------
            None
        """

        self.datastore_name = datastore_name
        self.ra = ra
        self.dec = dec
        self.search_cone = 2.0 * u.deg
        self.source_name = source_name
        self.tstart = tstart
        self.tstop = tstop
        self.model_name = None
        self.point_like = point_like
        self.obs_ids = obs_ids
        self.e_min = 0.1
        self.e_max = 30

        if scratch_path is None:
            self.scratch_path = Path(base_path + str(uuid.uuid4()))
        else:
            print ("Scratch path is set")
            self.scratch_path = Path(base_path + scratch_path)

        print ("Writing to ", self.scratch_path)
        self.scratch_path.mkdir(exist_ok=True)
        self.aeff_max = 10.0

    def initialize_analysis(self):
        """
        Sets up datastore and analysis paths

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        self.setup_datastore()
        self.setup_analysis()

    def setup_datastore(self):
        """
        Setup the datastore

        Creates a datastore from the inital directory passed. Gets the observation indices and observations.

        Parameters:
        -----------
            None

        Returns:
        --------
            Noce
        """
        self.datastore = DataStore.from_dir(self.datastore_name)
        if self.obs_ids is None:
            self.obs_ids = query_datastore(
                self.datastore_name,
                self.ra,
                self.dec,
                self.search_cone,
                self.tstart,
                self.tstop,
            )

        required_irf = "point-like" if self.point_like else "full-enclosure"
        self.observations = self.datastore.get_observations(
            self.obs_ids, required_irf=required_irf
        )

    def setup_fov(self):
        """
        Generate the exclusion regions for the FoV

        Queries known stars and gamma-ray sources within the FoV.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        excl = get_exclusion_regions(self.ra, self.dec)
        self.on_exclusion_region = CircleSkyRegion(
            center=SkyCoord(self.ra, self.dec, unit="deg", frame="icrs"),
            radius=0.4 * u.deg,
        )

        self.regions = [self.on_exclusion_region]
        for exc in excl:

            self.regions.append(
                CircleSkyRegion(
                    center=SkyCoord(exc[0], exc[1], unit="deg", frame="icrs"),
                    radius=u.Quantity(exc[2]),
                )
            )

            geom = WcsGeom.create(
                npix=(150, 150),
                binsz=0.05,
                skydir=self.target_position,
                proj="TAN",
                frame="icrs",
            )
            self.exclusion_mask = ~geom.region_mask(self.regions)

    def setup_analysis(self):
        """
        Setups the initial gammapy analysis.

        Creates required axes, geometries and dataset makers. This uses reflected region backgorund maker.

        Parameters:
        -----------
            None

        Returns:
        --------
            None

        """
        self.target_position = SkyCoord(
            ra=self.ra, dec=self.dec, unit="deg", frame="icrs"
        )

        self.setup_fov()

        energy_axis = MapAxis.from_energy_bounds(
            self.e_min, self.e_max, nbin=10, per_decade=True, unit="TeV", name="energy"
        )
        energy_axis_true = MapAxis.from_energy_bounds(
            0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
        )

        self.datasets = Datasets()

        # Loop over the observations
        for obs_id, observation in zip(self.obs_ids, self.observations):

            # Define the on region using rad_max from the observation
            if self.point_like:
                self.on_region = CircleSkyRegion(
                    center=self.target_position, radius=Angle(observation.rad_max.data[0][0] * u.deg)
                )
            # Otherwise use a fixed radius
            else:
                self.on_region = CircleSkyRegion(
                    center=self.target_position, radius=Angle("0.08944272 deg")
                )

            geom = RegionGeom.create(region=self.on_region, axes=[energy_axis])
            dataset_empty = SpectrumDataset.create(
                geom=geom, energy_axis_true=energy_axis_true
            )

            countainment_correction = False if self.point_like else True
            use_region_center = (True if self.point_like else False,)

            dataset_maker = SpectrumDatasetMaker(
                containment_correction=countainment_correction,
                selection=["counts", "exposure", "edisp"],
                use_region_center=use_region_center,
            )
            bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=self.exclusion_mask)
            safe_mask_masker = SafeMaskMaker(
                methods=["aeff-max"], aeff_percent=self.aeff_max
            )


            dataset = dataset_maker.run(
                dataset_empty.copy(name=str(obs_id)), observation
            )
            dataset_on_off = bkg_maker.run(dataset, observation)
            dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
            self.datasets.append(dataset_on_off)

        self.info_table = self.datasets.info_table(cumulative=True)
        # Stacked dataset used for spectral analysis
        self.dataset_stacked = Datasets(self.datasets).stack_reduce()

        path = self.scratch_path.joinpath("spectrum_analysis")
        path.mkdir(exist_ok=True)
        for dataset in self.datasets:
            dataset.write(filename=path / f"obs_{dataset.name}.fits.gz", overwrite=True)

    def setup_model(
        self, model_type: Optional[str] = "pwl", model_name: Optional[str] = None
    ):
        """
        Sets up the spectral model for the analysis.

        Parameters:
        -----------
            model_type : The type of the spectral model. Can be "pwl" (power law), "lp" or "logpar" or "logparbola" (log parabola),
                                    "expcut" or "cutoff" or "expontntial cutoff" (exponential cutoff). Defaults to "pwl".
            model_name : The name of the model. If None, the source name will be used.

        Raises:
            ValueError: If the model_type is not found in the incorporated models.

        Returns:
        --------
            None
        """

        pwl_models = ["pwl", "powerlaw"]
        logparabola_models = ["lp", "logpar", "logparbola"]
        exp_cut_models = ["expcut", "cutoff", "expontntial cutoff"]

        incorperated_models = pwl_models + logparabola_models + exp_cut_models
        if model_type.lower() in pwl_models:
            spectral_model = PowerLawSpectralModel(
                amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
                index=2,
                reference=1 * u.TeV,
            )

            spectral_model.parameters["index"].min = 1.0
            spectral_model.parameters["index"].max = 5.0
            spectral_model.parameters["amplitude"].min = 1e-19
            spectral_model.parameters["amplitude"].max = 1e-6

        elif model_type.lower() in logparabola_models:
            spectral_model = LogParabolaSpectralModel(
                amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
                alpha=2,
                beta=0.0,
                reference=1 * u.TeV,
            )
        elif model_type.lower() in exp_cut_models:
            spectral_model = ExpCutoffPowerLawSpectralModel(
                amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
                gamma=2,
                lambda_=0.0,
                reference=1 * u.TeV,
            )
        else:
            raise ValueError(
                f"Model type: {model_type} not found in \n\t{incorperated_models}"
            )

        if self.model_name is None:
            self.model_name = self.source_name
        else:
            self.model_name = model_name

        self.model = SkyModel(spectral_model=spectral_model, name=self.model_name)
        self.datasets.models = self.model
        self.dataset_stacked.models = self.model

    def fit_spectrum(self) -> SkyModel:
        """
        Fits the spectral model to the stacked dataset.

        The method runs the fit on the stacked dataset and then updates the models of the datasets with the results of the fit.

        Parameters:
        -----------
            None

        Returns:
        --------
            SkyModel: A copy of the sky model after the fit.

        """
        fit = Fit()
        fit.run(datasets=self.dataset_stacked)
        self.datasets.models = self.dataset_stacked.models.copy()
        self.model_best = self.dataset_stacked.models.copy()
        return self.dataset_stacked.models.copy()

    def get_spectral_points(
        self, e_min: float = 0.1, e_max: float = 10, n_bins: int = 10
    ) -> FluxPoints:
        """
        Estimates the flux points of the fitted model.

        Parameters:
        -----------
            e_min : The minimum energy for the flux points in TeV. Defaults to 0.1.
            e_max : The maximum energy for the flux points in TeV. Defaults to 10.
            n_bins : The number of energy bins for the flux points. Defaults to 10.

        Returns:
        --------
            FluxPoints: The estimated flux points.
        """
        energy_edges = np.geomspace(e_min, e_max, n_bins) * u.TeV

        fpe = FluxPointsEstimator(
            energy_edges=energy_edges, source=self.model_name, selection_optional="all"
        )
        self.flux_points = fpe.run(datasets=self.dataset_stacked)
        return self.flux_points

    def run_lightcurve(
        self,
        duration: float,
        tstart: (Optional[Time]) = None,
        tstop: (Optional[Time]) = None,
        e_min: Optional[float] = None,
        e_max: Optional[float] = None,
    ) -> FluxPoints:
        """
        Runs a light curve estimation.

        Parameters:
        -----------
            duration (float): The duration of each time bin for the light curve in days.
            tstart (Optional[Time]): The start time for the light curve. If None, the start time of the analysis will be used.
            tstop (Optional[Time]): The stop time for the light curve. If None, the stop time of the analysis will be used.
            e_min (float): The minimum energy for the light curve in TeV. Defaults to 0.1 TeV.
            e_max (float): The maximum energy for the light curve in TeV. Defaults to 30 TeV.

        Returns:
        --------
            LightCurve: The estimated light curve.
        """
        if tstart is None:
            tstart = self.tstart
        if tstop is None:
            tstop = self.tstop
        if e_min is None:
            e_min =  self.e_min
        if e_max is None:
            e_max = self.e_max

        t0 = tstart.mjd
        n_time_bins = ceil((tstop.mjd - tstart.mjd) / duration)
        times = t0 + np.arange(n_time_bins) * duration
        time_intervals = [
            Time([_tstart, _tstop], format="mjd")
            for _tstart, _tstop in zip(times[:-1], times[1:])
        ]

        norm = {
            "scan_min" : 0.01, 
            "scan_max" : 100,
            "scan_n_values" : 500
        }

        lc_maker_1d = LightCurveEstimator(
            time_intervals=time_intervals,
            energy_edges=[e_min, e_max] * u.TeV,
            source=self.source_name,
            reoptimize=False,
            selection_optional="all",
            norm=norm
        )
        self.datasets.models = self.model_best
        self.lc_1d = lc_maker_1d.run(self.datasets)
        return self.lc_1d

    def summarize_results(
        self,
        time_scales: Optional[List[float]] = None,
        fname: str = "results.fits",
        e_min: float = 0.1,
        e_max: float = 30,
    ) -> None:
        """
        Summarizes the results of the analysis.

        Parameters:
        -----------
            time_scales: List of time scales for the light curve. Defaults to None.
            fname: str, optional, default "results.fits", the name of the output file.
            e_min: float, optional, default 0.1, minimum energy [TeV] for the flux points
            e_max: float, optional, default 30, maximum energy [TeV] for the flux points

        Returns:
        --------
            None
        """

        hdu_list = [fits.PrimaryHDU()]
        titles = ["PRIMARY"]

        # 1 info table
        hdu_list.append(fits.BinTableHDU(self.info_table))
        titles.append("INFO")
        # 2 model
        hdu_list.append(fits.BinTableHDU(self.model.parameters.to_table()))
        titles.append("MODEL")
        # 3 Spectral points
        hdu_list.append(fits.BinTableHDU(self.flux_points.to_table()))
        titles.append("FLUX_POINTS")
        # 4 Light curves

        for time_scale in time_scales:
            try:
                lc = self.run_lightcurve(time_scale, e_min=e_min, e_max=e_max)
                hdu_list.append(fits.BinTableHDU(lc.to_table(format="lightcurve")))
                titles.append(f"LIGHTCURVE_{time_scale}")
            except:
                print(f"Light curve for time scale {time_scale} failed.")

        hdul = fits.HDUList(hdu_list)
        for i, title in enumerate(titles):
            hdul[i].header["EXTNAME"] = title
            hdul[i].name = title

        hdul.writeto(self.scratch_path.joinpath(fname), overwrite=True)
