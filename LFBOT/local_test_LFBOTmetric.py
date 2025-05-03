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
# Local Utility: Uniform sky injection
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

# --------------------------------------------
# Light Curve Model for LFBOTs
# --------------------------------------------
class LFBOT_LC:
    """
    Generate synthetic light curves for Luminous Fast Blue Optical Transients (LFBOTs).

    Light curves are modeled with band-dependent rise and fade slopes, peaking around ~1 day,
    and spanning a fast-evolving timescale of ~0.1 to 10 days. Only g and r bands are populated,
    consistent with the predominantly blue emission of LFBOTs.
    """
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None):
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded LFBOT templates from {load_from}")
            return

        self.data = []
        self.filts = ["g", "r"]  # Only g and r used
        self.t_grid = np.logspace(-1, 1, num_samples)  # 0.1 to 10 days

        rng = np.random.default_rng(42)
        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                m0 = rng.uniform(-21.5, -20)  # peak mag
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
        if lc_indx >= len(self.data):
            lc_indx = len(self.data) - 1
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --------------------------------------------------
# Light Curve Template Generator (Separate from Population)
# --------------------------------------------------
def generateLFBOTTemplates(
    num_samples=100, num_lightcurves=1000,
    save_to="LFBOT_templates.pkl"
):
    """
    Generate synthetic LFBOT light curve templates and save to file.

    Templates are modeled with rapid rise and fade timescales,
    and only g and r bands are populated, consistent with LFBOT properties.
    """
    if os.path.exists(save_to):
        print(f"Found existing LFBOT templates at {save_to}. Not regenerating.")
        return

    lc_model = LFBOT_LC(num_samples=num_samples, num_lightcurves=num_lightcurves, load_from=None)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data}, f)
    print(f" Saved synthetic LFBOT light curve templates to {save_to}")

# --------------------------------------------
# Base Metric for LFBOTs
# --------------------------------------------
class BaseLFBOTMetric(BaseMetric):
    """
    Base metric class for evaluating LFBOT light curves against simulated observations.

    This class handles light curve interpolation, extinction correction, and signal-to-noise
    calculation, providing a standardized evaluation framework for derived LFBOT metrics.
    """
    def __init__(self, metricName='BaseLFBOTMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 filter_include=None, load_from="LFBOT_templates.pkl",
                 lc_model=None, **kwargs):

        if lc_model is not None:
            self.lc_model = lc_model
        else:
            self.lc_model = LFBOT_LC(load_from=load_from)

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
            if f not in self.lc_model.filts:
                # Skip filters without light curve templates (e.g., i, z, y)
                continue
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

# --------------------------------------------
# Detection Metric for LFBOTs
# --------------------------------------------
class LFBOT_GRBStyleDetectMetric(BaseLFBOTMetric):
    """
    Apply GRB afterglow-style detection logic to LFBOTs:
    - Option A: ≥2 detections in same filter ≥30 min apart
    - Option B: ≥2 epochs ≥30 min apart with second epoch detected in ≥2 filters
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'LFBOT_GRBStyleDetect')
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

        # Option A: ≥2 detections in the same filter ≥30 minutes apart
        for f in np.unique(filters):
            mask = filters == f
            if np.sum(snr[mask] >= 5) >= 2:
                if np.ptp(times[mask]) >= 0.5 / 24:
                    detected = True
                    break

        # Option B: ≥2 epochs with second detected in ≥2 filters
        if not detected:
            t_detect = times[snr >= 5]
            if len(t_detect) > 0:
                if len(np.unique(filters[snr >= 5])) >= 2:
                    if np.ptp(t_detect) >= 0.5 / 24:
                        detected = True

        if detected:
            detected_mask = snr >= 5
            obs_record['detected'] = detected_mask
            self.latest_obs_record = obs_record

            first_det_mjd = np.nan
            last_det_mjd = np.nan
            rise_time = np.nan
            fade_time = np.nan

            if np.any(detected_mask):
                first_det_mjd = obs_record['mjd_obs'][detected_mask].min()
                last_det_mjd = obs_record['mjd_obs'][detected_mask].max()
                rise_time = first_det_mjd - (self.mjd0 + slice_point['peak_time'])
                fade_time = last_det_mjd - (self.mjd0 + slice_point['peak_time'])

            peak_index = np.argmin(obs_record['mag_obs'])
            peak_mjd = obs_record['mjd_obs'][peak_index]
            peak_mag = obs_record['mag_obs'][peak_index]

            obs_record.update({
                'first_det_mjd': first_det_mjd,
                'last_det_mjd': last_det_mjd,
                'rise_time_days': rise_time,
                'fade_time_days': fade_time,
                'sid': slice_point['sid'],
                'file_indx': slice_point['file_indx'],
                'ra': slice_point['ra'],
                'dec': slice_point['dec'],
                'distance_Mpc': slice_point['distance'],
                'peak_mjd': peak_mjd,
                'peak_mag': peak_mag,
                'ebv': slice_point['ebv'],
            })

            self.obs_records[slice_point['sid']] = obs_record
            return 1.0

        else:
            self.latest_obs_record = None
            return 0.0
# --------------------------------------------
# Characterization Metric for LFBOTs
# --------------------------------------------
class LFBOTCharacterizeMetric(BaseLFBOTMetric):
    """
    Given the provided scientific context, we define a minimal photometric characterization
    criterion for Rubin LSST observations of Luminous Fast Blue Optical Transients (LFBOTs).

    Based on the science description:
    - Full confirmation of LFBOT nature requires external follow-up (radio, X-ray, or spectroscopy),
      as noted explicitly in the provided science case.
    - Optical surveys like Rubin primarily serve to detect candidates and monitor fast fading behavior.
    - Example events like AT2018cow and AT2023fhn demonstrate ~0.2 mag/day fading rates
      and durations at high luminosity of less than 10–12 days.
    - You indicated that specific filters (g and r bands) dominate, and monitoring fading tails is
      considered helpful, even if it does not constitute definitive classification.

    Therefore, we define photometric characterization as:
    - Having at least 4 detections with SNR ≥3,
    - Spanning a timespan of at least 3 days.

    These limits ensure that Rubin can constrain the rapid evolution of LFBOT candidates in optical light,
    sufficient to inform and trigger multi-wavelength follow-up, even though true physical classification
    depends on external datasets.

    This structure mirrors the GRB afterglow characterization metric but is relaxed:
    - No ≥3 filters condition is required (because LFBOTs are primarily blue and concentrated in g and r).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_lc(dataSlice, slice_point, return_full_obs=True)

        good = snr >= 3
        if np.sum(good) < 4:
            return 0.0
        duration = np.ptp(times[good])

        if duration >= 3:
            return 1.0
        return 0.0
        

# --------------------------------------------
# LFBOT Population Rate
# --------------------------------------------
def sample_lfbot_rate(t_start, t_end, d_min, d_max, rate_density=50e-9):
    """
    Estimate the number of LFBOT events expected in the survey window.

    Calculates the number of events by multiplying the volumetric LFBOT rate
    by the comoving volume between the specified distance bounds (d_min, d_max),
    and the duration of the simulated survey in years.

    Parameters
    ----------
    t_start, t_end : float
        Start and end times of the survey window (in days).
    d_min, d_max : float
        Minimum and maximum luminosity distances (in Mpc).
    rate_density : float
        Volumetric LFBOT event rate in units of events per Mpc^3 per year.

    Returns
    -------
    int
        Poisson-sampled number of LFBOT events expected over the survey period.
    """
    years = (t_end - t_start) / 365.25
    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)
    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)

# --------------------------------------------
# LFBOT Population Slicer Generator
# --------------------------------------------
def generateLFBOTPopSlicer(t_start=1, t_end=3652, seed=42,
                           d_min=10, d_max=1000, num_lightcurves=1000,
                           gal_lat_cut=None, load_from=None, save_to=None):
    """
    Generate a synthetic population of LFBOT events across the sky.

    Events are distributed uniformly over the celestial sphere, assigned random distances,
    peak times, and matched to synthetic light curve templates. Galactic extinction is applied
    using the SFD dust map. Optionally saves or loads populations from a pickle file.

    Parameters
    ----------
    t_start, t_end : float
        Start and end times of the simulated survey window (in days).
    d_min, d_max : float
        Minimum and maximum luminosity distances (in Mpc).
    seed : int
        Random number generator seed for reproducibility.
    gal_lat_cut : float or None
        Minimum Galactic latitude (deg) to exclude crowded plane regions, if specified.
    load_from : str or None
        Path to load existing population pickle file.
    save_to : str or None
        Path to save newly generated population pickle file.
    """
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded LFBOT population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    n_events = sample_lfbot_rate(t_start, t_end, d_min, d_max)

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

    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved LFBOT population to {save_to}")

    return slicer

