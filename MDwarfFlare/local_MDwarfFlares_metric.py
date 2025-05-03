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

# --------------------------------------------------
# Utility: Convert Galactic to Equatorial coordinates
# --------------------------------------------------
def equatorialFromGalactic(lon, lat):
    gal = Galactic(l=lon * u.deg, b=lat * u.deg)
    equ = gal.transform_to(ICRSFrame())
    return equ.ra.deg, equ.dec.deg

# -----------------------------------------------------------------------------
# 1. Local Utility: Uniform sky injection
# -----------------------------------------------------------------------------
def uniform_sphere_degrees(n_points, seed=None):

    """
    Generate RA, Dec uniformly over the celestial sphere.

    Parameters
    ----------
    n_points : int
        Number of sky positions.
    seed : int or None
        Random seed.

    Returns
    -------
    ra : ndarray
        Right Ascension in degrees.
    dec : ndarray
        Declination in degrees.
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n_points)
    z = rng.uniform(-1, 1, n_points)  # uniform in cos(theta)
    dec = np.degrees(np.arcsin(z))   # arcsin(z) gives uniform in solid angle
    print("YAY! UNIFORM SPHERE!")
    return ra, dec

# ------------------------------------------------------
# 2. Quiescent Magnitude Distributions (UltraCoolSheet)
# ------------------------------------------------------
QUIESCENT_MAG_MEANS = {
    'g': 18.025,
    'r': 16.698,
    'i': 14.262,
    'z': 12.936,
    'y': 12.104
}

# ------------------------------------------------------
# 3. Light Curve Model
# ------------------------------------------------------
class MDwarfFlareLC:
    def __init__(self, num_samples=100, delta_mag=5.0):
        self.data = []
        self.filts = ["g", "r", "i", "z", "y"]
        self.delta_mag = delta_mag
        rng = np.random.default_rng(42)

        # Rise and fade rates
        self.rise_rates = {'g': (0.009, 3.89), 'r': (0.005, 2.15), 'i': (0.0027, 0.415),
                           'z': (0.002, 0.113), 'y': (0.0014, 0.051)}
        self.fade_rates = {'g': (0.006, 0.79), 'r': (0.003, 0.44), 'i': (0.0019, 0.085),
                           'z': (0.001, 0.023), 'y': (0.001, 0.01)}

        t_quiescent = np.linspace(-1.0, -0.05, num_samples // 5)
        t_rise = np.linspace(-0.05, 0, num_samples // 5)
        t_fade = np.linspace(0.01, 1.5, num_samples)

        for _ in range(100):
            flare = {}
            for f in self.filts:
                quiescent = QUIESCENT_MAG_MEANS[f]
                peak_mag = quiescent - delta_mag
                rise = rng.uniform(*self.rise_rates[f])
                fade = rng.uniform(*self.fade_rates[f])

                mag_quiescent = np.full_like(t_quiescent, quiescent)
                mag_rise = peak_mag - rise * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
                mag_peak = np.full((1,), peak_mag)
                mag_fade = peak_mag + fade * (np.log10(1 + t_fade))

                t_full = np.concatenate([t_quiescent, t_rise, [0], t_fade])
                mag_full = np.concatenate([mag_quiescent, mag_rise, mag_peak, mag_fade])

                flare[f] = {'ph': t_full, 'mag': mag_full}
            self.data.append(flare)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t,
                         self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# ------------------------------------------------------
# 4. Base MDwarfFlare Metric
# ------------------------------------------------------
class BaseMDwarfFlareMetric(BaseMetric):
    def __init__(self, metric_name='BaseMDwarfFlareMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 lc_model=None, **kwargs):
        if lc_model is not None:
            self.lc_model = lc_model
        else:
            self.lc_model = MDwarfFlareLC()

        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metric_name, units='Detection Efficiency', badval=badval, **kwargs)

    def evaluate_flare(self, dataSlice, slice_point, return_full_obs=True):
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size)

        for f in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']

        snr = m52snr(mags, dataSlice[self.m5Col])
        filters = dataSlice[self.filterCol]
        times = t

        if return_full_obs:
            obs_record = {
                'mjd_obs': dataSlice[self.mjdCol],
                'mag_obs': mags,
                'snr_obs': snr,
                'filter': filters,
            }
            return snr, filters, times, obs_record
        return snr, filters, times

# ------------------------------------------------------
# 5. Detection Metric
# ------------------------------------------------------
class MDwarfFlareDetectMetric(BaseMDwarfFlareMetric):
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_flare(dataSlice, slice_point, return_full_obs=True)

        detected_5sigma = snr >= 5
        if np.sum(detected_5sigma) < 1:
            return 0.0
        if np.ptp(times[detected_5sigma]) > 1.5:
            return 0.0  # Must be <1.5 days between detections
        return 1.0

# ------------------------------------------------------
# 6. Classical vs Complex Characterization Metric
# ------------------------------------------------------
class MDwarfFlareCharacterizeMetric(BaseMDwarfFlareMetric):
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_flare(dataSlice, slice_point, return_full_obs=True)

        significant = snr >= 0.5
        if np.sum(significant) < 4:
            return 0.0

        peaks = snr >= 1.5
        if np.sum(peaks) >= 2 and np.ptp(times[peaks]) >= 0.1:
            return 1.0  # Complex flare
        return 0.5  # Classical flare

# ------------------------------------------------------
# 7. Population Injection Generator
# ------------------------------------------------------
def generateMDwarfFlarePop(t_start=1, t_end=3652, seed=42,
                           rate_deg2_hr=1.0, gal_lat_cut=30, max_events=1_000_000,
                           save_to=None, load_from=None, n_lightcurves=100):
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded M Dwarf flare population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    hours = (t_end - t_start) * 24
    n_events = int(rate_deg2_hr * 18000 * hours)
    n_events = min(n_events, max_events)

    ra, dec = uniform_sphere_degrees(n_events, seed=seed)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    if gal_lat_cut:
        gal_b = coords.galactic.b.deg
        mask = np.abs(gal_b) < gal_lat_cut
        ra, dec = ra[mask], dec[mask]
        coords = coords[mask]

    peak_times = rng.uniform(t_start, t_end, len(ra))
    file_indx = rng.integers(0, n_lightcurves, len(ra))
    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['distance'] = np.full(len(ra), 1.0)
    slicer.slice_points['ebv'] = ebv_vals

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved M Dwarf flare population to {save_to}")

    return slicer
