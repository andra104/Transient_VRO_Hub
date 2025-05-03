from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues
from rubin_sim.maf.metric_bundles import MetricBundle
from rubin_sim.maf.utils import m52snr
import matplotlib.pyplot as plt
from rubin_sim.maf.db import ResultsDb
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from rubin_sim.maf.slicers import HealpixSlicer
import rubin_sim.maf.metric_bundles as metric_bundles
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from rubin_sim.maf.metrics import CountMetric
from rubin_sim.maf.maps import StellarDensityMap
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
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
# Local Utility: Uniform sky injection
# -----------------------------------------------------------------------------
def uniformSphere(n_points, seed=None):
    """
    Generate points uniformly distributed on the surface of a sphere.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    ra : ndarray
        Right Ascension values in degrees.
    dec : ndarray
        Declination values in degrees.
    """
    rng = np.random.default_rng(seed)
    eps = 1e-10
    u = rng.uniform(eps, 1 - eps, n_points)
    v = rng.uniform(eps, 1 - eps, n_points)

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    ra = np.degrees(theta)
    dec = 90 - np.degrees(phi)
    return ra, dec

# -----------------------------------------------------------------------------
# Light Curve Parameter Definitions for Shock-Cooling Emission Peak in SNe IIb
# -----------------------------------------------------------------------------
SCE_PARAMETERS = {
    'g': {
        'rise_rate_mu': 1.09,
        'rise_rate_sigma': 0.34,
        'fade_rate_mu': 0.23,
        'fade_rate_sigma': 0.087,
        'peak_mag_min': -18.65,
        'peak_mag_max': -14.82,
        'duration_at_peak': 2.35,
        'second_peak_mag_range': (-17.5, -15.0),
        'second_peak_rise_mu': 0.082,
        'second_peak_rise_sigma': 0.059
    },
    'r': {
        'rise_rate_mu': 0.97,
        'rise_rate_sigma': 0.35,
        'fade_rate_mu': 0.18,
        'fade_rate_sigma': 0.095,
        'peak_mag_min': -18.21,
        'peak_mag_max': -14.82,
        'duration_at_peak': 2.90,
        'second_peak_mag_range': (-17.9, -15.4),
        'second_peak_rise_mu': 0.091,
        'second_peak_rise_sigma': 0.053
    }
}

# ============================================================
# Light Curve Generator for Shock Cooling Events
# ============================================================
class ShockCoolingLC:
    def __init__(self, num_samples=100, load_from=None):
        """
        Generate or load synthetic light curves for Shock Cooling Emission in SN IIb.

        Parameters
        ----------
        num_samples : int
            Number of time samples per light curve
        load_from : str or None
            If provided, loads templates from a pickle file
        """
        # --- Load pre-generated templates if requested ---
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.durations = data['durations']
            self.filts = list(self.data[0].keys())
            print(f"Loaded {len(self.data)} shock cooling light curves from {load_from}")
            return

        # --- Otherwise generate templates from scratch ---
        self.data = []
        self.durations = {}
        self.filts = list(SCE_PARAMETERS.keys())

        def sample_rate(mu, sigma):
            return np.random.normal(mu, sigma)

        t_rise = np.linspace(-1.5, 0, num_samples // 5)
        t_fade = np.linspace(0.01, 5, num_samples)
        t_rerise = np.linspace(7, 13, num_samples)

        for _ in range(100):
            lightcurve = {}
            for f in self.filts:
                params = SCE_PARAMETERS[f]

                peak_mag_1 = np.random.uniform(params['peak_mag_min'], params['peak_mag_max'])
                rise1 = sample_rate(params['rise_rate_mu'], params['rise_rate_sigma'])
                fade1 = sample_rate(params['fade_rate_mu'], params['fade_rate_sigma'])
                mag_rise = peak_mag_1 - rise1 * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
                mag_peak1 = np.full((1,), peak_mag_1)
                mag_fade = peak_mag_1 + fade1 * t_fade

                peak_mag_2 = np.random.uniform(*params['second_peak_mag_range'])
                rise2 = sample_rate(params['second_peak_rise_mu'], params['second_peak_rise_sigma'])
                mag_rerise = peak_mag_2 - rise2 * (13 - t_rerise) / 6

                t_full = np.concatenate([t_rise, [0], t_fade, t_rerise])
                mag_full = np.concatenate([mag_rise, mag_peak1, mag_fade, mag_rerise])
                lightcurve[f] = {'ph': t_full, 'mag': mag_full}

                if f not in self.durations:
                    self.durations[f] = {'rise': [], 'fade': [], 'rerise': []}
                self.durations[f]['rise'].append(np.ptp(t_rise))
                self.durations[f]['fade'].append(np.ptp(t_fade))
                self.durations[f]['rerise'].append(np.ptp(t_rerise))

            self.data.append(lightcurve)




    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)


# ============================================================
# Shared Evaluation Logic and Metric Subclasses
# ============================================================
class BaseShockCoolingType2bMetric(BaseMetric):
    """
    Base class for evaluating Shock-Cooling Emission Peaks in Type IIb SNe.
    Simulates light curves and applies detection and classification criteria.
    """
    def __init__(self, metricName='ShockCoolingType2bMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', ptsNeeded=1,
                 mjd0=59853.5, outputLc=False, badval=-666, include_second_peak=True, load_from="ShockCooling_templates.pkl", **kwargs):

        self._lc_model = ShockCoolingLC(load_from=load_from)
        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.outputLc = outputLc
        self.mjd0 = mjd0
        self.include_second_peak = include_second_peak  # Enables second peak detection logic


        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName,
                         units='Detected, 0 or 1', maps=['DustMap'],
                         badval=badval, **kwargs)

    def characterize_sce(self, snr, times, filtername, file_indx):

        idx = np.where(snr >= 0.5)[0]
        if len(idx) < 6:
            return 'uncharacterized'
    
        dur_rise   = self._lc_model.durations[filtername]['rise'][file_indx]
        dur_fade   = self._lc_model.durations[filtername]['fade'][file_indx]
        dur_rerise = self._lc_model.durations[filtername]['rerise'][file_indx]
        
        rise  = np.sum((times >= -dur_rise) & (times <= 0) & (snr >= 0.5))
        fade  = np.sum((times > 0) & (times <= dur_fade) & (snr >= 0.5))
        rerise = np.sum((times > dur_fade) & (times <= dur_fade + dur_rerise) & (snr >= 0.5))


        if rise >= 2 and fade >= 2 and rerise >= 1: #changed from 2 
            return 'classical'
        elif rise + fade + rerise >= 6:
            return 'ambiguous'
        return 'uncharacterized'

    def evaluate_sce(self, dataSlice, slice_point):
        """
        Evaluate whether a Shock-Cooling light curve is detected, characterized, and shows a second peak.
    
        Parameters
        ----------
        dataSlice : numpy.recarray
            The subset of the cadence data relevant for this slice point.
        slice_point : dict
            Metadata for the specific injected SN event (distance, time, etc.).
    
        Returns
        -------
        result : dict
            Contains:
            - 'detection' (int): 1 if detected, 0 otherwise
            - 'characterization' (str): 'classical', 'ambiguous', or 'uncharacterized'
            - 'double_peak' (bool): True if second peak is resolved (optional)
        """
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size, dtype=float)
    
        for f in np.unique(dataSlice[self.filterCol]):
            if f not in self._lc_model.filts:
                continue
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self._lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']  # Apply extinction
    
        snr = m52snr(mags, dataSlice[self.m5Col])
    
        # Detection logic
        detected = 0
        for f in self._lc_model.filts:
            filt_mask = (dataSlice[self.filterCol] == f)
            t_filt = t[filt_mask]
            snr_filt = snr[filt_mask]
            if np.sum(snr_filt >= 5) >= 2:
                if np.any(np.diff(np.sort(t_filt[snr_filt >= 5])) >= 0.5):
                    detected = 1
                    break
    
        # Default values for not-detected events
        characterization = 'uncharacterized'
        double_peak_detected = False
    
        if detected:
            characterization = self.characterize_sce(snr, t, f, slice_point['file_indx'])
            if self.include_second_peak:
                for f in self._lc_model.filts:
                    mask = (dataSlice[self.filterCol] == f)
                    t_filt = t[mask]
                    snr_filt = snr[mask]
                    second_rise = np.sum((t_filt > 7) & (t_filt <= 13) & (snr_filt >= 0.5))
                    if second_rise >= 2:
                        double_peak_detected = True
                        break

        return {
            'detection': detected,
            'characterization': characterization,
            'double_peak': double_peak_detected
        }



# Metric Subclasses
class ShockCoolingType2bDetectMetric(BaseShockCoolingType2bMetric):
    def run(self, dataSlice, slice_point=None):
        return self.evaluate_sce(dataSlice, slice_point)['detection']

class ShockCoolingType2bCharacterizeMetric(BaseShockCoolingType2bMetric):
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'classical' else 0

class ShockCoolingType2bClassicalMetric(BaseShockCoolingType2bMetric):
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'classical' else 0

class ShockCoolingType2bAmbiguousMetric(BaseShockCoolingType2bMetric):
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'ambiguous' else 0


class ShockCoolingType2bUncharacterizedMetric(BaseShockCoolingType2bMetric):
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result['characterization'] == 'uncharacterized' else 0
        
class ShockCoolingType2bDoublePeakMetric(BaseShockCoolingType2bMetric):
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_sce(dataSlice, slice_point)
        if result['detection'] == 0:
            return self.badval
        return 1 if result.get('double_peak', False) else 0

# Updated utility for realistic uniform sky with WFD mask
def uniform_wfd_sky(n_points, mask_map, nside=64, seed=None):
    rng = np.random.default_rng(seed)
    ipix_all = np.arange(len(mask_map))
    ipix_wfd = ipix_all[mask_map > 0.5]  # Use mask threshold to define footprint
    selected_ipix = rng.choice(ipix_wfd, size=n_points, replace=True)
    theta, phi = hp.pix2ang(64, selected_ipix, nest=False)
    dec = 90 - np.degrees(theta)
    ra = np.degrees(phi)
    return ra, dec


# ============================================================
# Population Generator for Shock-Cooling Type IIb Events
# ============================================================
def generateShockCoolingType2bSlicer(prob_map, nside=64, t_start=1, t_end=3652,
                                     seed=42, rate_per_year=65000,
                                     d_min=10, d_max=300,
                                     gal_lat_cut=None, save_to=None):
    """
    Generate slicer populated with simulated Shock Cooling SNe IIb events.
    Ensures declination values are within [-90, +90] and no NaNs are passed to healpy.
    """
    n_years = (t_end - t_start) / 365.25
    n_points = int(rate_per_year * n_years)
    print(f"Generating {n_points} SN IIb events from rate: {rate_per_year}/yr × {n_years:.2f} yr")

    # Draw HEALPix pixels based on LSST footprint probability
    rng = np.random.default_rng(seed)
    hp_indices = rng.choice(len(prob_map), size=n_points, p=prob_map)
    theta, phi = hp.pix2ang(nside, hp_indices, nest=False)
    dec = 90.0 - np.degrees(theta)
    ra = np.degrees(phi)

    # Apply optional Galactic latitude cut
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        mask = np.abs(b) > gal_lat_cut
        coords = coords[mask]

    # Simulate distances, peak times, template indices
    distances = rng.uniform(d_min, d_max, len(coords))
    peak_times = rng.uniform(t_start, t_end, len(coords))
    file_indx = rng.integers(0, 100, len(coords))

    # Query extinction
    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    valid = (~np.isnan(coords.ra.deg) &
             ~np.isnan(coords.dec.deg) &
             ~np.isnan(distances) &
             ~np.isnan(peak_times) &
             ~np.isnan(file_indx) &
             ~np.isnan(ebv_vals))

    ra = coords.ra.deg[valid]
    dec = coords.dec.deg[valid]
    distances = distances[valid]
    peak_times = peak_times[valid]
    file_indx = file_indx[valid]
    ebv_vals = ebv_vals[valid]

    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    slicer.slice_points['ra'] = ra
    slicer.slice_points['dec'] = dec
    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['sid'] = hp.ang2pix(nside, np.radians(90. - dec), np.radians(ra), nest=False)
    slicer.slice_points['nside'] = nside
    slicer.slice_points['dec_rad'] = np.radians(dec)

    if save_to:
        slice_data = dict(slicer.slice_points)
        with open(save_to, 'wb') as f:
            pickle.dump(slice_data, f)
        print(f"Saved Shock Cooling population to {save_to}")

    # Save light curve templates
    lc_model = ShockCoolingLC()
    with open("ShockCooling_templates.pkl", "wb") as f:
        pickle.dump({'lightcurves': lc_model.data, 'durations': lc_model.durations}, f)
    print("Saved synthetic SCE light curve templates to ShockCooling_templates.pkl")


    return slicer

    return slicer