# ==========================================
# TEMPLATE: UNIVERSAL TRANSIENT METRIC BASE
# ==========================================

import numpy as np
import pickle
import os
import healpy as hp
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18 as cosmo, z_at_value
from astropy.coordinates import SkyCoord, Galactic, ICRS as ICRSFrame
from dustmaps.sfd import SFDQuery

from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from rubin_sim.phot_utils import DustValues
from rubin_sim.maf.utils import m52snr

# =========================================================
# Generic Population Functions 
# =========================================================

# Utility: Uniform sphere sampling (RA, Dec in degrees)
def uniform_sphere_degrees(n_points, seed=None):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n_points)
    z = rng.uniform(-1, 1, n_points)
    dec = np.degrees(np.arcsin(z))
    return ra, dec

# Utility: Sample number of events based on volumetric rate
def sample_transient_rate(t_start, t_end, d_min, d_max, rate_density):
    years = (t_end - t_start) / 365.25
    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)
    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)

# =========================================================
# LIGHT CURVE TEMPLATE CLASS (TO BE CUSTOMIZED PER TRANSIENT)
# =========================================================
class YourTransientLC:
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None):
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded light curves from {load_from}")
            return

        self.data = []
        self.filts = ["u", "g", "r", "i", "z", "y"]
        self.t_grid = np.logspace(-1, 2, num_samples)

        rng = np.random.default_rng(42)
        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                # CUSTOMIZE HERE
                m0 = rng.uniform(-18, -16)  # Example for SNe
                alpha_rise = rng.uniform(-1.0, -0.5)
                alpha_fade = rng.uniform(0.5, 1.5)
                t0 = 1.0
                mag = np.where(
                    self.t_grid < t0,
                    m0 + 2.5 * alpha_rise * np.log10(self.t_grid / t0),
                    m0 + 2.5 * alpha_fade * np.log10(self.t_grid / t0)
                )
                lc[f] = {'ph': self.t_grid, 'mag': mag}
            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        if lc_indx >= len(self.data):
            lc_indx = len(self.data) - 1
        return np.interp(t, self.data[lc_indx][filtername]['ph'], self.data[lc_indx][filtername]['mag'], left=99, right=99)


# =============================================
# BASE METRIC CLASS
# =============================================
class BaseYourTransientMetric(BaseMetric):
    def __init__(self, metricName='BaseYourTransientMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="YourTransient_templates.pkl", **kwargs):

        self.lc_model = YourTransientLC(load_from=load_from)
        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc
        self.filter_include = filter_include

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detection Efficiency', badval=badval, **kwargs)

    def evaluate_lc(self, dataSlice, slice_point, return_full_obs=True):
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size)
        for f in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lc_model.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']
            mags[infilt] += 5 * np.log10(slice_point['distance'] * 1e6) - 5

        snr = m52snr(mags, dataSlice[self.m5Col])
        filters = dataSlice[self.filterCol]
        times = t

        if return_full_obs:
            obs_record = {'mjd_obs': dataSlice[self.mjdCol], 'mag_obs': mags, 'snr_obs': snr, 'filter': filters}
            return snr, filters, times, obs_record
        return snr, filters, times


# =============================================
# DETECTION METRIC CLASS (Customize Logic)
# =============================================
class YourTransientDetectMetric(BaseYourTransientMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_lc(dataSlice, slice_point, return_full_obs=True)

        if self.filter_include is not None:
            keep = np.isin(filters, self.filter_include)
            snr = snr[keep]
            filters = filters[keep]
            times = times[keep]
            for k in ['mjd_obs', 'mag_obs']:
                obs_record[k] = obs_record[k][keep]

        detected = False

        # --- CUSTOMIZE your detection logic below ---
        for f in np.unique(filters):
            mask = filters == f
            if np.sum(snr[mask] >= 5) >= 2:
                if np.ptp(times[mask]) >= 0.5 / 24:
                    detected = True
                    break

        if detected:
            detected_mask = snr >= 5
            obs_record['detected'] = detected_mask
            obs_record.update({
                'first_det_mjd': np.min(obs_record['mjd_obs'][detected_mask]) if np.any(detected_mask) else np.nan,
                'last_det_mjd': np.max(obs_record['mjd_obs'][detected_mask]) if np.any(detected_mask) else np.nan,
                'sid': slice_point['sid'],
                'file_indx': slice_point['file_indx'],
                'ra': slice_point['ra'],
                'dec': slice_point['dec'],
                'distance_Mpc': slice_point['distance'],
                'ebv': slice_point['ebv']
            })
            self.obs_records[slice_point['sid']] = obs_record
            return 1.0

        return 0.0

# ===========================================================================
# Possible List of Metric subclasses 
#like Characterize to push detect through - dependent on paramters provided.
# ===========================================================================

class TransientCharacterizeMetric(BaseGRBAfterglowMetric):
    ...
        return 0.0

# --------------------------------------------------
# TRANSIENT POPULATION GENERATOR TEMPLATE
# --------------------------------------------------


# Main Population Generator
def generateTransientPopSlicer(
    t_start=1, t_end=3652, seed=42,
    d_min=10, d_max=1000,
    rate_density=1e-8,
    num_lightcurves=1000,
    gal_lat_cut=None,
    load_from=None,
    save_to=None
):
    """
    Generate a synthetic transient population slicer.
    
    Parameters
    ----------
    gal_lat_cut : float or None
        Minimum Galactic latitude to avoid Milky Way plane.
    load_from : str or None
        Path to load a saved population pickle.
    save_to : str or None
        Path to save the generated population pickle.
    """

    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    n_events = sample_transient_rate(t_start, t_end, d_min, d_max, rate_density)

    ra, dec = uniform_sphere_degrees(n_events, seed=seed)
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
        ra, dec = ra[mask], dec[mask]
        distances = distances[mask]
        peak_times = peak_times[mask]
        file_indx = file_indx[mask]
        ebv_vals = ebv_vals[mask]
        coords = coords[mask]

    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved transient population to {save_to}")

    return slicer
