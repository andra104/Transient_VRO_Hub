from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

from rubin_sim.maf.utils import m52snr
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
from astropy.cosmology import z_at_value
import numpy as np
import glob
import os

import pickle 

__all__ = [
    'LFBOT_LC',
    'generateLFBOTTemplates',
    'BaseLFBOTMetric',
    'LFBOTDetectMetric',
    'LFBOTCharacterizeMetric',
    'generateLFBOTPopSlicer'
]

# --------------------------------------------
# Uniform Sphere Healpix
# --------------------------------------------
def inject_uniform_healpix(nside, n_events, seed=42):
    """Generate RA, Dec from uniformly sampled Healpix pixels."""
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    pix = rng.choice(npix, size=n_events, replace=True)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = np.degrees(0.5 * np.pi - theta)
    return ra, dec

# --------------------------------------------
# Light Curve Model
# --------------------------------------------
class LFBOT_LC:
    """Generate synthetic light curves for LFBOTs with rapid evolution in g and r bands."""
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None):
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            return

        self.data = []
        self.filts = ["g", "r"]
        self.t_grid = np.logspace(-1, 1, num_samples)

        rng = np.random.default_rng(42)
        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                m0 = rng.uniform(-21.5, -20)
                alpha_rise = rng.uniform(-2.5, -0.25)
                alpha_fade = rng.uniform(0.15, 0.45)
                t0 = 1.0
                mag = np.where(
                    self.t_grid < t0,
                    m0 + 2.5 * alpha_rise * np.log10(self.t_grid / t0),
                    m0 + 2.5 * alpha_fade * np.log10(self.t_grid / t0)
                )
                lc[f] = {'ph': self.t_grid, 'mag': mag}
            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        """Interpolate magnitude at time t for a given filter and light curve index."""
        if lc_indx >= len(self.data):
            lc_indx = len(self.data) - 1
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --------------------------------------------
# Template Generator
# --------------------------------------------
def generateLFBOTTemplates(num_samples=100, num_lightcurves=1000, save_to="LFBOT_templates.pkl"):
    """Generate and save LFBOT light curve templates."""
    if os.path.exists(save_to):
        print(f"Found existing LFBOT templates at {save_to}. Not regenerating.")
        return

    lc_model = LFBOT_LC(num_samples=num_samples, num_lightcurves=num_lightcurves)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data}, f)
    print(f"Saved synthetic LFBOT light curve templates to {save_to}")

# --------------------------------------------
# Base Metric
# --------------------------------------------
class BaseLFBOTMetric(BaseMetric):
    """Base metric for evaluating LFBOT light curves against LSST simulated observations."""
    def __init__(self, metricName='BaseLFBOTMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, badval=-666, filter_include=None,
                 load_from="LFBOT_templates.pkl", lc_model=None, **kwargs):

        self.lc_model = lc_model if lc_model is not None else LFBOT_LC(load_from=load_from)
        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.filter_include = filter_include

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detection Efficiency', badval=badval, **kwargs)

    def evaluate_lc(self, dataSlice, slice_point, return_full_obs=True):
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size)
    
        for f in np.unique(dataSlice[self.filterCol]):
            if (self.filter_include is not None and f not in self.filter_include) or f not in self.lc_model.filts:
                continue
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']
            mags[infilt] += 5 * np.log10(slice_point['distance'] * 1e6) - 5
    
        snr = m52snr(mags, dataSlice[self.m5Col])
        filters = dataSlice[self.filterCol]
        times = t
    
        if return_full_obs:
            obs_record = {
                'mjd_obs': dataSlice[self.mjdCol],
                'mag_obs': mags,
                'snr_obs': snr,
                'filter': filters
            }
            return snr, filters, times, obs_record
        return snr, filters, times


# --------------------------------------------
# Detection Metric
# --------------------------------------------
class LFBOTDetectMetric(BaseLFBOTMetric):
    """
    LFBOT detection metric based on multiple observational criteria.

    An event is considered detected if **either** of the following conditions is satisfied:

    Option A (Color + Host Redshift Trigger):
        - At least 2 detections (SNR ≥ 5) occur in a single epoch (within 30 minutes).
        - These detections must span ≥2 different filters (e.g., g and r).
        - This provides both color and time-coincident data to estimate luminosity from redshift.

    Option B (Duration + Luminosity Trigger):
        - At least 3 detections (SNR ≥ 5) occur in different epochs (i.e., different nights or >30 minutes apart).
        - The detections must span ≤6 days.
        - This enables light curve duration estimation (rise/fade timescale).
    
    Additional Constraints:
        - All detections must have SNR ≥ 5.
        - Each detection must be separated by ≥30 minutes to rule out moving object contamination.

    These criteria are adapted from known LFBOT light curves (e.g., AT2018cow, CSS161010),
    and consistent with LSST cadence sampling and ZTF detection heuristics.
    """
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_lc(dataSlice, slice_point, return_full_obs=True)
        good = snr >= 5
        if np.sum(good) < 2:
            return 0.0

        # Unique observation times for good detections (rounded to nearest 30 min)
        unique_times = np.unique(np.round(times[good] * 48) / 48)

        # Option A: At least one epoch with 2 filters within 30 minutes
        for t in unique_times:
            idx = np.where(good & (np.abs(times - t) < 0.02))[0]
            if len(idx) >= 2:
                unique_filters = np.unique(filters[idx])
                if len(unique_filters) >= 2:
                    return 1.0

        # Option B: At least 3 detections across different epochs within ≤6 days
        if len(unique_times) >= 3 and (np.max(unique_times) - np.min(unique_times)) <= 6:
            return 1.0

        return 0.0

class LFBOTSingleFilterDetectMetric(BaseLFBOTMetric):
    """Detectability in a single filter: ≥1 detection with SNR ≥ 5."""
    def run(self, dataSlice, slice_point=None):
        snr, filters, times = self.evaluate_lc(dataSlice, slice_point, return_full_obs=False)
        good = snr >= 5
        if np.any(good):
            return 1.0
        return 0.0


# --------------------------------------------
# Characterization Metric
# --------------------------------------------
class LFBOTCharacterizeMetric(BaseLFBOTMetric):
    """Photometric characterization: ≥4 SNR≥3 detections spanning ≥3 days."""
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, _ = self.evaluate_lc(dataSlice, slice_point, return_full_obs=True)

        good = snr >= 3
        if np.sum(good) < 4:
            return 0.0
        duration = np.ptp(times[good])
        if duration >= 3:
            return 1.0
        return 0.0

# --------------------------------------------
# Population Generator
# --------------------------------------------
def generateLFBOTPopSlicer(t_start=1, t_end=3652, seed=42,
                           d_min=10, d_max=1000, num_lightcurves=1000,
                           gal_lat_cut=None, nside=64,
                           load_from=None, save_to=None):
    """Generate synthetic population of LFBOT events with extinction and distance."""
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        return slicer

    rng = np.random.default_rng(seed)

    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)
    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    rate_density = (50e-9)*1000
    years = (t_end - t_start) / 365.25
    n_events = rng.poisson(rate_density * V * years)

    ra, dec = inject_uniform_healpix(nside, n_events, seed=seed)
    dec = np.clip(dec, -89.9999, 89.9999)
    dec_rad = np.radians(dec)

    slicer = UserPointsSlicer(ra=ra, dec=dec_rad, badval=0)
    slicer.slice_points['ra'] = ra
    slicer.slice_points['dec'] = dec_rad

    distances = rng.uniform(d_min, d_max, n_events)
    peak_times = rng.uniform(t_start, t_end, n_events)
    file_indx = rng.integers(0, num_lightcurves, n_events)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        mask = np.abs(b) > gal_lat_cut
        for key in ['ra', 'dec', 'distance', 'peak_time', 'file_indx', 'ebv']:
            slicer.slice_points[key] = slicer.slice_points[key][mask]

    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)

    return slicer


