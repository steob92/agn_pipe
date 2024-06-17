import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chi2
from typing import Tuple, Optional, List
from astropy.table import Table


def get_variability_index(tab: Table) -> Tuple[float, int]:
    """Compute the variability index of a set of light curves.

    Parameters:
    -----------
    tab_lc: astropy Table of light curve measurements includeing likelihood profiles


    Returns:
    --------
    ts_var: float Variability index
    ndf: int Number of degrees of freedom

    """

    local_max = []
    norm_range = np.linspace(0.201, 4.999, 1000)
    total_prof = np.zeros(len(norm_range))
    for i in range(len(tab)):
        inter = interp1d(
            tab[i]["norm_scan"][0], tab[i]["stat_scan"][0], kind="quadratic"
        )

        # Get the maximum of the interpolated profile
        local_max.append(np.min(inter(norm_range)))
        # Record the total
        total_prof += inter(norm_range)

    # Likelihood of the global maximum (minimum -2 logl)
    global_max = np.min(total_prof)
    # Get the likelihood of the free maximum (minimum -2 logl)
    free_max = np.sum(local_max)
    # Variability index
    ts_var = global_max - free_max
    # ndf = n_points - 1
    return ts_var, len(tab) - 1


def get_variability_probability(ts_var: float, ndf: int) -> float:
    """Compute the probability of the variability index.

    Parameters:
    -----------
    ts_var: float Variability index
    ndf: int Number of degrees of freedom

    Returns:
    --------
    prob: float Probability of the variability index

    """
    prob = chi2.sf(ts_var, ndf)
    return prob


def get_change_points(tab: Table, threshold: Optional[float] = 0.005) -> List[int]:
    """Compute the change points of a set of light curves.

    Parameters:
    -----------
    tab: astropy Table of light curve measurements includeing likelihood profiles
    threshold: float, optional, default 0.005, threshold chi^2 probability for a change point

    Returns:
    --------
    change_points: List of integers, change points of the light curve

    """

    n_obs = len(tab)
    last_change_point = 0
    change_points = [0]

    while last_change_point < n_obs:

        change_point = None
        for i in range(last_change_point, n_obs):
            # Get the variablity index and probability for this block
            ts_var, ndf, _flux, _flux_errl, _flux_erru = get_variability_index(
                tab[last_change_point : i + 1]
            )
            prob = get_variability_probability(ts_var, ndf)

            # Check if the block shows variability
            if prob < threshold:
                change_points.append(i)
                change_point = i
                last_change_point = i
                break

        # If no change point is found, break the loop
        # This should happen for a constant source or at the end of the light curve
        if change_point is None:
            break

    change_points.append(n_obs - 1)
    return change_points
