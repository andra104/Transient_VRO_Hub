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
import numpy as np
import glob
import os

import pickle 

# --- TEMPORARY print limiter for debugging complex flares ---
_complex_flare_counter = 0
_MAX_COMPLEX_FLARES_TO_PRINT = 10


def equatorialFromGalactic(lon, lat):
    gal = Galactic(l=lon * u.deg, b=lat * u.deg)
    equ = gal.transform_to(ICRSFrame())  
    return equ.ra.deg, equ.dec.deg

#local - uniformSphere import
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


class MDwarfFlareLC:
    def __init__(self, num_samples=100, delta_mag=5):
        self.data = []
        self.filts = ["g", "r", "i", "z", "y"]
        self.delta_mag = delta_mag  # store it


        # Light curve parameters
        # Rise rates (mag/day)
        self.rise_rates = {
            'g': (0.009, 3.89), 'r': (0.005, 2.15),
            'i': (0.0027, 0.415), 'z': (0.002, 0.113),
            'y': (0.0014, 0.051)
        }
        # Fade rates (mag/day)
        self.fade_rates = {
            'g': (0.006, 0.79), 'r': (0.003, 0.44),
            'i': (0.0019, 0.085), 'z': (0.001, 0.023),
            'y': (0.001, 0.01)
        }
        # Peak absolute magnitude range
        self.peak_mag_range = {
            'g': (14.05, 16.44), 'r': (14.01, 15.99),
            'i': (12.90, 15.21), 'z': (12.13, 14.06),
            'y': (11.68, 13.18)
        }

        def sample_rate(rate_range):
            mu, sigma = np.mean(rate_range), (rate_range[1] - rate_range[0]) / 3
            return np.clip(np.random.normal(mu, sigma), *rate_range)

        t_quiescent = np.linspace(-1.0, -0.05, num_samples // 5)
        t_rise = np.linspace(-0.05, 0, num_samples // 5)
        t_fade = np.linspace(0.01, 1.5, num_samples)


        for _ in range(100):
            flare = {}
            for f in self.filts:
                peak_mag = np.random.uniform(*self.peak_mag_range[f])
                rise = sample_rate(self.rise_rates[f])
                fade = sample_rate(self.fade_rates[f])
                mag_quiescent = np.full_like(t_quiescent, peak_mag + self.delta_mag)  # 5 mag below peak 
                mag_rise = peak_mag - rise * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
                mag_peak = np.full((1,), peak_mag)
                mag_fade = peak_mag + fade * (np.log10(1 + t_fade))
                
                t_full = np.concatenate([t_quiescent, t_rise, [0], t_fade])
                mag_full = np.concatenate([mag_quiescent, mag_rise, mag_peak, mag_fade])

                flare[f] = {'ph': t_full, 'mag': mag_full}
            self.data.append(flare)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --- Utility function to check time span and number of detections ---
def passes_time_span_constraint(times, max_span_days=None, min_points=1):
    """
    Check if the detection times pass the configured span and point-count thresholds.

    Parameters
    ----------
    times : ndarray
        Array of detection times.
    max_span_days : float or None
        Maximum allowable time span for detections (None disables check).
    min_points : int
        Minimum number of detection points.

    Returns
    -------
    bool
        True if constraints are met.
    """
    if times.size < min_points:
        return False
    if max_span_days is not None:
        if (np.max(times) - np.min(times)) > max_span_days:
            return False
    return True


class BaseMDwarfFlareMetric(BaseMetric):
    """Base class for simulating and evaluating M Dwarf flare light curves and detection criteria."""

    def __init__(self, metricName='MDwarfFlareMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', ptsNeeded=1,
                 mjd0=59853.5, outputLc=False, badval=-666,
                 resolved=True, **kwargs):

        self.resolved = resolved
        self._flare_cache = {}
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.outputLc = outputLc
        self.mjd0 = mjd0
        self.ax1 = DustValues().ax1
    
        # Extract and remove delta_mag from kwargs
        delta_mag = kwargs.pop('delta_mag', 5.0)
        self.lightcurves = MDwarfFlareLC(delta_mag=delta_mag)
    
        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super().__init__(col=cols, metric_name=metricName, 
                         units='Detected, 0 or 1', maps=maps, 
                         badval=badval, **kwargs)

    def characterize_flare(self, snr, t, mags):
        # A flare is 'classical' if it has a single peak,
        # and 'complex' if it has multiple peaks separated by ≥0.1 days.
        # If fewer than 4 points above 0.5σ, it is 'uncharacterized'.
        idx = np.where(snr >= 0.5)[0]
        if len(idx) < 4:
            return 'uncharacterized'
        significant = np.where(snr >= 1.5)[0]
        if len(significant) < 2:
            return 'classical'
        t_peaks = np.sort(t[significant])
        if np.any(np.diff(t_peaks) >= 0.1):
            return 'complex'
        return 'classical'

    def evaluate_flare(self, dataSlice, slice_point):
        sid = int(slice_point['sid'])
    
        # Use cache to avoid re-characterizing the same SID
        if sid in self._flare_cache:
            return self._flare_cache[sid]
    
        # Compute relative time from flare peak
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
        mags = np.zeros(t.size, dtype=float)
    
        # Interpolate synthetic light curve for each observed filter
        for f in np.unique(dataSlice[self.filterCol]):
            if f not in self.lightcurves.filts:
                continue
            infilt = np.where(dataSlice[self.filterCol] == f)
            mags[infilt] = self.lightcurves.interp(t[infilt], f, slice_point['file_indx'])
            mags[infilt] += self.ax1[f] * slice_point['ebv']  # Apply dust extinction
    
        snr = m52snr(mags, dataSlice[self.m5Col])
        result = {'detection': 0, 'characterization': 'none'}
    
        if self.resolved:
            # Original criterion:
            # Resolved detection required >=1 detection at ≥5σ or 3 at ≥3σ
            # AND duration between detections must be ≤0.5 days (same night)
            detected_5sig = (snr >= 5) #5σ
            detected_3sig = (snr >= 3) #3σ
            detection_times = t[detected_3sig | detected_5sig]
    
            # New criterion: allow configurable time span (e.g., 5 days) and points, originally 0.5 
            if passes_time_span_constraint(detection_times, max_span_days=5, min_points=1): 
                if np.sum(detected_5sig) >= 1 or np.sum(detected_3sig) >= 1:
                    result['detection'] = 1
                    result['characterization'] = self.characterize_flare(snr, t, mags)
                    self._flare_cache[sid] = result
                    return result
    
        else:
            # Original criterion:
            # Unresolved detection required ≥2 points at ≥5σ
            # AND any two detections separated by >15 minutes (to avoid moving objects)
            det = np.where(snr >= 5)[0]  # Step 1: Get all ≥5σ detections
            if det.size >= 2:            # Step 2: Ensure at least 2 such detections exist
                tdet = np.sort(t[det])   # Step 3: Sort their times
                if np.any(np.diff(tdet) > (15 / 60 / 24)):  # Step 4: Check if any two are >15 min apart. 15 min divided by 60 min divided by 24 hours.
                    result['detection'] = 1

            # Unresolved flares are not reliably characterizable
            if result['detection'] == 1:
                result['characterization'] = 'uncharacterized'
    
        self._flare_cache[sid] = result
        return result




# ---- NEW METRIC SUBCLASSES ----

class MDwarfFlareDetectionMetric(BaseMDwarfFlareMetric):
    """Metric to detect whether a resolved M Dwarf flare is observable."""
    
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_flare(dataSlice, slice_point)
        return result['detection']

class MDwarfFlareClassicalMetric(BaseMDwarfFlareMetric):
    """Metric to count resolved classical (single-peaked) M Dwarf flares."""
    
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_flare(dataSlice, slice_point)
        return 1 if result['characterization'] == 'classical' else 0

class MDwarfFlareComplexMetric(BaseMDwarfFlareMetric):
    """Metric to count resolved complex (multi-peaked) M Dwarf flares."""
    
    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_flare(dataSlice, slice_point)
        return 1 if result['characterization'] == 'complex' else 0

class MDwarfFlareUnresolvedMetric(BaseMDwarfFlareMetric):
    """Metric to detect transient flares with no visible quiescent component (unresolved)."""
    
    def __init__(self, **kwargs):
        super().__init__(resolved=False, **kwargs)

    def run(self, dataSlice, slice_point=None):
        result = self.evaluate_flare(dataSlice, slice_point)
        return result['detection']

class MDwarfFlareResolvedMetric(BaseMDwarfFlareMetric):
    """Metric to count all resolved M Dwarf flares (regardless of classical or complex)."""

    def run(self, dataSlice, slice_point=None):
        if not self.resolved:
            return 0
        result = self.evaluate_flare(dataSlice, slice_point)
        return result['detection']

def generateMDwarfFlareSlicer(t_start=1, t_end=3652, seed=42, n_files=100,
                              gal_lat_cut=None,
                              load_from=None, save_to=None,
                              rate_deg2_hr=None, max_events=None):

    """
    M Dwarf flare population generator.
    
    Parameters
    ----------
    t_start : float
        Start time in survey days (default=1).
    t_end : float
        End time in survey days (default=3652).
    seed : int
        RNG seed.
    n_files : int
        Number of light curve templates.
    gal_lat_cut : float or None
        If set, apply a galactic latitude cut (|b| < value).
    load_from : str or None
        If given and the file exists, load the slicer from this pickle file.
    save_to : str or None
        If set, save the generated slicer to this pickle file.
    rate_deg2_hr : float
        Flare occurrence rate in deg⁻² hr⁻¹. (default=1.0)
    """
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded flare population from {load_from}")
        return slicer

    np.random.seed(seed)
    hours = (t_end - t_start) * 24
    max_events = 2000000
    n_events_raw = int(rate_deg2_hr * 18000 * hours)
    n_events = min(n_events_raw, max_events) if max_events is not None else n_events_raw


    # Inject uniformly across the sky
    ra, dec = uniformSphere(n_events, seed=seed)

    # Apply Galactic latitude cut if requested
    if gal_lat_cut is not None:
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        gal_b = coords.galactic.b.deg
        mask = np.abs(gal_b) < gal_lat_cut
        ra, dec = ra[mask], dec[mask]
        print(f"Applied Galactic latitude cut: |b| < {gal_lat_cut} deg")

    n_selected = len(ra)
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_selected)
    file_indx = np.random.randint(0, n_files, size=n_selected)

    # Compute realistic dust extinction values    
    # Query SFD dust map
    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['distance'] = np.full(n_selected, 1.0)  # distance-independent
    slicer.slice_points['ebv'] = ebv_vals  # extinction now added realistically

    if save_to:
        slice_data = {
            'ra': ra,
            'dec': dec,
            'peak_time': peak_times,
            'file_indx': file_indx,
            'distance': np.full(n_selected, 1.0),
            'ebv': ebv_vals
        }
        with open(save_to, 'wb') as f:
            pickle.dump(slice_data, f)
        print(f"Saved flare population to {save_to}")

    return slicer