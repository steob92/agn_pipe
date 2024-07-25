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
    scan_min = tab[0]["norm_scan"][0].min()
    scan_max = tab[0]["norm_scan"][0].max()
    
    norm_range = np.linspace(scan_min, scan_max, 1000)
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
    probs = []
    last_prob = 0
    while last_change_point < n_obs:

        change_point = None
        for i in range(last_change_point, n_obs):
            # Get the variablity index and probability for this block
            ts_var, ndf = get_variability_index(
                tab[last_change_point : i + 1]
            )
            prob = get_variability_probability(ts_var, ndf)

            # Check if the block shows variability
            if prob < threshold:
                change_points.append(i)
                change_point = i
                last_change_point = i
                probs.append(last_prob)
                break
            last_prob = prob
        # If no change point is found, break the loop
        # This should happen for a constant source or at the end of the light curve
        if change_point is None:
            break

    probs.append(last_prob)
    change_points.append(n_obs - 1)
    return change_points, probs



def get_bottom_up(scans: List[np.ndarray]) -> float:
    """Get the likelihood of the bottom-up model. This assumes no variability and that all data points are drawn from the same distribution.

    Parameters:
    -----------
    scans: List of numpy arrays, likelihood profiles of the light curves

    Returns:
    --------
    logl: float, likelihood of the bottom-up model

    """
    total_prof = np.zeros(len(scans[0]))
    # Minimum of the sum of the profiles
    for i in range(len(scans)):
        total_prof += scans[i]
    return total_prof.min()

def get_top_down(scans : List[np.ndarray]) -> float:
    """Get the likelihood of the top-down model. This assumes that each data point is drawn from a different distribution.

    Parameters:
    -----------
    scans: List of numpy arrays, likelihood profiles of the light curves

    Returns:
    --------
    logl: float, likelihood of the top-down model

    """

    logl = 0
    # Sum the minimum of each profile
    for i in range(len(scans)):
        logl += scans[i].min()

    return logl

def get_pair_likelihood(scans : List[np.ndarray]) -> List[float]:
    """Get the likelihood of the pair model. This assumes that each pair of data points is drawn from the same distribution.
    
    Parameters:
    -----------
    scans: List of numpy arrays, likelihood profiles of the light curves
    
    Returns:
    --------
    pairs: List of floats, likelihood of the pair model for each pair of data points
    
    """
    pairs = []
    for i in range(len(scans)-1):
        # Likelihood of combining the two profiles
        prof = (scans[i] + scans[i+1]).min()
        # Likelihood of keeping the two profiles separate
        free = scans[i].min() + scans[i+1].min()
        # Return the difference between the two
        # This allows us to find the pair that gives the smallest decrease in the model quality
        pairs.append(prof - free)
    return pairs

def get_scans(tab_lc : Table) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Extract the likelihood profiles of the light curves.
    
    Parameters:
    -----------
    tab_lc: astropy Table of light curve measurements
    
    Returns:
    --------
    norm_range: numpy array, normalized range of the likelihood profiles
    scans: List of numpy arrays, likelihood profiles of the light curves
    
    """

    # Adapt the range of the likelihood profiles
    norm_min = tab_lc[0]["norm_scan"][0].min()
    norm_max = tab_lc[0]["norm_scan"][0].max()

    # Interpolate the likelihood profiles
    norm_range = np.linspace(norm_min, norm_max, 1000)
    scans = []
    for i in range(len(tab_lc)):
        inter = interp1d(tab_lc[i]["norm_scan"][0], tab_lc[i]["stat_scan"][0])
        scans.append(inter(norm_range))
    
    return norm_range, scans


def get_edges(tab_lc : Table) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Get the edges of the light curves.
    
    Parameters:
    -----------
    tab_lc: astropy Table of light curve measurements
    
    Returns:
    --------
    likelihood: numpy array, likelihood of the model for each number of edges
    dof: numpy array, degrees of freedom for each number of edges
    edges: List of List of integers, edges of the light curves
    
    """

    # Extract the scans
    norm_range, scans = get_scans(tab_lc)

    # Get the number of bins
    n_bins = len(tab_lc)
    dof = np.arange(1,len(tab_lc)+1)
    
    # Get the likelihood of the models
    likelihood = np.zeros(n_bins)
    likelihood[-1] = get_bottom_up(scans)
    likelihood[0] = get_top_down(scans)

    # Get the edges
    current_edges = [ i for i in range (len(tab_lc))]
    edges = [current_edges[::]]
    
    # Make a copy of the likelihood scans
    tmp_scans = scans[::]
    # Loop over the number of potential binnings
    for i in range(1,n_bins-1):

        # Get the likelihood of the pair model
        logl_pair = get_pair_likelihood(tmp_scans)
        # Find the pair that gives the smallest decrease in the model quality
        amin = np.argmin(logl_pair)
        # Get the likelihood of the pair model
        adj = tmp_scans.pop(amin+1)

        # Pop out the edge from the list and save new model
        current_edges.pop(amin+1)
        edges.append(current_edges[::])
        
        # Update the likelihood to reflect the addition of the new edge
        tmp_scans[amin] += adj
        # Record the likelihood of the model
        likelihood[i] = np.sum([tmp.min() for tmp in tmp_scans])


    # Add the bottom up model
    edges.append([0, len(tab_lc)-1])
    
    # Reverse the dof 
    return likelihood, dof[::-1], edges    