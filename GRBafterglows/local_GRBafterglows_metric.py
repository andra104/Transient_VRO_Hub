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
    
    plt.figure(figsize=(8, 4))
    plt.scatter(ra, dec, s=1, alpha=0.3, label="Injected", color="black")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("GRB Sky UniformSphere Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("YAY! UNIFORM SPHERE!")
    return ra, dec


# --------------------------------------------
# Power-law GRB afterglow model based on Zeh et al. (2005)
# --------------------------------------------
class GRBAfterglowLC:
    """
    Simulate GRB afterglow light curves using a power-law model.

    Light curves follow:
        m(t) = m_0 + 2.5 * alpha * log10(t/t_0)
    where alpha is the temporal slope (rise or decay), t is time (days),
    and m_0 is the peak magnitude (from Zeh et al. 2005).

    The rise slope is negative (brightening), and the decay is positive (fading).
    """
    def __init__(self, num_samples=100, num_lightcurves=1000, load_from=None):
        """
        Parameters
        ----------
        num_samples : int
            Number of time points to sample in the light curve (log-uniformly spaced).
        load_from : str or None
            If provided and valid, loads light curve templates from a pickle file.
        """
        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                data = pickle.load(f)
            self.data = data['lightcurves']
            self.filts = list(self.data[0].keys())
            print(f"Loaded GRB afterglow templates from {load_from}")
            return

        self.data = []
        self.filts = ["u", "g", "r", "i", "z", "y"]
        self.t_grid = np.logspace(-1, 2, num_samples)  # 0.1 to 100 days

        decay_slope_range = (0.5, 2.5)
        rise_slope_range = (-1.5, -0.5)
        peak_mag_range = (-24, -22)

        rng = np.random.default_rng(42)
        for _ in range(num_lightcurves):
            lc = {}
            for f in self.filts:
                m0 = rng.uniform(*peak_mag_range)
                alpha_rise = rng.uniform(*rise_slope_range)
                alpha_fade = rng.uniform(*decay_slope_range)
                t0 = 0.3
                mag = np.where(
                    self.t_grid < t0,
                    m0 + 2.5 * alpha_rise * np.log10(self.t_grid / t0),
                    m0 + 2.5 * alpha_fade * np.log10(self.t_grid / t0)
                )
                lc[f] = {'ph': self.t_grid, 'mag': mag}
            self.data.append(lc)

    def interp(self, t, filtername, lc_indx=0):
        """
        Interpolate the light curve for the given filter and index at times `t`.

        Parameters
        ----------
        t : array_like
            Times relative to peak (days).
        filtername : str
            LSST filter name (u, g, r, i, z, y).
        lc_indx : int
            Index of the light curve in the template set.

        Returns
        -------
        magnitudes : array_like
            Interpolated magnitudes, clipped at 99 for out-of-range.
        """
        if lc_indx >= len(self.data):
            print(f"Warning: lc_indx {lc_indx} out of bounds, using last template.")
            lc_indx = len(self.data) - 1
        return np.interp(t,
                         self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)

# --------------------------------------------------
# Light Curve Template Generator (Separate from Population)
# --------------------------------------------------
def generateGRBAfterglowTemplates(
    num_samples=100, num_lightcurves=1000,
    save_to="GRBAfterglow_templates.pkl"
):
    """
    Generate synthetic GRB afterglow light curve templates and save to file.
    """
    if os.path.exists(save_to):
        print(f"Found existing GRB afterglow templates at {save_to}. Not regenerating.")
        return

    lc_model = GRBAfterglowLC(num_samples=num_samples, num_lightcurves=num_lightcurves, load_from=None)
    with open(save_to, "wb") as f:
        pickle.dump({'lightcurves': lc_model.data}, f)
    print(f"Saved synthetic GRB light curve templates to {save_to}")


# --------------------------------------------
# Base GRB Metric with extinction and SNR
# --------------------------------------------
class BaseGRBAfterglowMetric(BaseMetric):
    def __init__(self, metricName='BaseGRBAfterglowMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 mjd0=59853.5, outputLc=False, badval=-666,
                 filter_include=None,
                 load_from="GRBAfterglow_templates.pkl",
                 lc_model=None,  # <-- NEW
                 **kwargs):
        """
        Parameters
        ----------
        lc_model : GRBAfterglowLC or None
            Shared GRB light curve model object. If None, load from file.
        """
        if lc_model is not None:
            self.lc_model = lc_model
        else:
            self.lc_model = GRBAfterglowLC(load_from=load_from)

        self.ax1 = DustValues().ax1  # From rubin_sim.phot_utils
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.outputLc = outputLc
        self.filter_include = filter_include

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName,
                         units='Detection Efficiency',
                         badval=badval, **kwargs)


    def evaluate_grb(self, dataSlice, slice_point, return_full_obs=True):
        """
        Evaluate GRB light curve at the location and time of the slice point.
        Apply extinction, distance modulus, and optional filter inclusion.
        """
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
            obs_record = {
                'mjd_obs': dataSlice[self.mjdCol],
                'mag_obs': mags,
                'snr_obs': snr,
                'filter': filters,
                # NO 'detected' YET -- will be set later if detected!
            }
            return snr, filters, times, obs_record
        return snr, filters, times



# --------------------------------------------
# Unified Detection metric
# --------------------------------------------
class GRBAfterglowDetectMetric(BaseGRBAfterglowMetric):
    """ 

    Option A: ≥2 detections in a single filter, ≥30 minutes apart
    
    Option B: ≥2 epochs, second has ≥2 filters; first can be a non-detection
    
    This is an “either/or” detection logic. 
    
    This event is detected if it passes either the intra-night multi-detection or the epoch-based detection criteria.
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'GRB_Detect')
        self.obs_records = {}  # <-- NEW: to store all detected event records individually

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        
        if self.filter_include is not None:
            keep = np.isin(filters, self.filter_include)
            snr = snr[keep]
            filters = filters[keep]
            times = times[keep]
            for k in ['mjd_obs', 'mag_obs']:
                obs_record[k] = obs_record[k][keep]

        # -------- Detection Logic --------
        
        detected = False
    
        # Option A: 2 detections in same filter ≥30min apart
        for f in np.unique(filters):
            mask = filters == f
            if np.sum(snr[mask] >= 5) >= 2:
                if np.ptp(times[mask]) >= 0.5 / 24:
                    detected = True
                    break
    
        # Option B: 2 filters ≥5σ ≥30min apart
        if not detected:
            t_detect = times[snr >= 5]
            if len(t_detect) > 0:
                if len(np.unique(filters[snr >= 5])) >= 2:
                    if np.ptp(t_detect) >= 0.5 / 24:
                        detected = True

        # -------- Save Detection Metadata --------
    
        if detected:
            detected_mask = snr >= 5
            obs_record['detected'] = (snr >= 5)
            self.latest_obs_record = obs_record

            # Calculate rise and fade times
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
        
            # Update obs_record with full metadata
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
        
            # Save this full event
            self.obs_records[slice_point['sid']] = obs_record
        
            self.latest_obs_record = obs_record
            return 1.0
        else:
            self.latest_obs_record = None
            return 0.0



# --------------------------------------------
# Characterization metric — extended multi-band follow-up
# --------------------------------------------
class GRBAfterglowCharacterizeMetric(BaseGRBAfterglowMetric):
    """
    Characterization metric for GRB Afterglows.

    This metric tests whether the transient can be sufficiently characterized for follow-up
    science goals. An event is considered 'characterized' if it meets two criteria:
    
    (1) At least 4 observations with signal-to-noise ratio (SNR) ≥ 3.
    (2) Among those detections, the observations span at least 3 different filters 
        and cover a duration of at least 3 days.

    These thresholds are motivated by the need to capture the transient's color evolution 
    and fading behavior across multiple bands and epochs, which are key for identifying
    and classifying GRB afterglows compared to other fast-evolving transients.
    
    This design ensures that events classified as 'characterized' have sufficient
    multi-band and temporal information to allow basic modeling and comparison to 
    theoretical GRB afterglow light curves.
    """
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        good = snr >= 3
        if np.sum(good) < 4:
            return 0.0
        n_filters = len(np.unique(filters[good]))
        duration = np.ptp(times[good])
        if n_filters >= 3 and duration >= 3:
            return 1.0
        return 0.0

# --------------------------------------------
# Spectroscopic Triggerability Metric
# Detects if ≥2 filters are triggered within 0.5 days of peak
# --------------------------------------------
class GRBAfterglowSpecTriggerableMetric(BaseGRBAfterglowMetric):
    """
    Spectroscopic triggerability metric for GRB Afterglows.

    This metric evaluates whether a GRB afterglow would be suitable for rapid spectroscopic follow-up.
    An event is considered triggerable if:

    (1) At least 2 different filters detect the event with SNR ≥ 5,
    (2) Both detections occur within 0.5 days (12 hours) after the light curve peak.

    The thresholds are motivated by the observational window when afterglows are brightest and most
    amenable to spectroscopy, as informed by studies of early GRB afterglow behavior (Zeh et al. 2005)
    and typical response times for ground-based spectroscopic facilities.

    Prioritizing early, multi-band detections ensures that spectra can be taken while the afterglow 
    remains bright enough for classification and redshift determination.
    """
    def __init__(self, **kwargs):
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)

        within_half_day = times <= 0.5
        early = (snr >= 5) & within_half_day

        if len(np.unique(filters[early])) >= 2:
            return 1.0
        return 0.0

# --------------------------------------------
# Color Evolution Metric
# Detects ≥2 epochs with multi-color information to constrain synchrotron cooling
# --------------------------------------------
class GRBAfterglowColorEvolveMetric(BaseGRBAfterglowMetric):
    """
    Color evolution detection metric for GRB Afterglows.

    This metric assesses whether an event shows measurable color (spectral) evolution over time.
    An event satisfies the criterion if:

    (1) At least 4 observations are detected with SNR ≥ 3,
    (2) These observations cluster into ≥ 2 distinct epochs (grouped at 0.5-day resolution),
    (3) Each epoch includes detections in at least 2 different filters.

    The detection of color evolution is critical for constraining synchrotron cooling breaks,
    energy injection episodes, and jet structure in GRB afterglows. These requirements
    are based on observational constraints for detecting chromatic breaks described in Zeh et al. (2005)
    and adapted to Rubin's cadence characteristics.

    By requiring multi-color detections across epochs, this metric distinguishes genuine 
    evolving afterglows from static or non-evolving fast transients.
    """
    def __init__(self, **kwargs):
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)

        detected = (snr >= 3)
        if np.sum(detected) < 4:
            return 0.0

        # Group by rounded times to cluster into epochs
        t_epoch = np.round(times[detected] * 2) / 2  # bin to 0.5-day resolution
        f_epoch = filters[detected]

        epoch_colors = {}
        for t, f in zip(t_epoch, f_epoch):
            epoch_colors.setdefault(t, set()).add(f)

        multi_color_epochs = [e for e in epoch_colors.values() if len(e) >= 2]

        if len(multi_color_epochs) >= 2:
            return 1.0
        return 0.0

# --------------------------------------------
# Historical Non-Detection Match Metric
# Checks whether the transient would stand out against deep coadds
# --------------------------------------------
class GRBAfterglowHistoricalMatchMetric(BaseGRBAfterglowMetric):
    """
    Archival non-detection match metric for GRB Afterglows.

    This metric checks whether the transient would stand out as a new source compared
    to deep archival imaging. An event passes if:

    (1) Any portion of its light curve is brighter than the assumed archival coadd depth 
        (default = 27.0 magnitudes).

    This logic is based on the expectation that GRB afterglows have no persistent
    optical counterparts prior to the event. The archival depth value reflects Rubin’s
    expected Wide-Fast-Deep coadded survey depth.

    This metric was designed to filter out background variable sources such as AGNs,
    variable stars, or other contaminants that could otherwise mimic a GRB-like transient
    in photometric detection pipelines.
    """
    def __init__(self, coaddDepth=27.0, **kwargs):
        """
        Parameters
        ----------
        coaddDepth : float
            Simulated archival limiting magnitude.
        """
        self.coaddDepth = coaddDepth
        super().__init__(load_from="GRBAfterglow_templates.pkl", **kwargs)

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = self.evaluate_grb(dataSlice, slice_point, return_full_obs=True)
        # Check if any detection is brighter than the archival depth
        mags = np.zeros(times.size)
        for f in np.unique(filters):
            mask = filters == f
            mags[mask] = self.lc_model.interp(times[mask], f, slice_point['file_indx'])
            mags[mask] += self.ax1[f] * slice_point['ebv']
            mags[mask] += 5 * np.log10(slice_point['distance'] * 1e6) - 5

        if np.any(mags < self.coaddDepth):
            return 1.0  # Would stand out above archival image
        return 0.0


# --------------------------------------------
# GRB volumetric rate model (on-axis ≈ 10⁻⁹ Mpc⁻³ yr⁻¹)
# --------------------------------------------
def sample_grb_rate_from_volume(t_start, t_end, d_min, d_max, rate_density=1e-8): #1e-8 to account for dirty fireball and off axis, 1e-9 without
    """
    Estimate the number of GRBs from comoving volume and volumetric rate.

    Parameters
    ----------
    t_start : float
        Start of the time window (days).
    t_end : float
        End of the time window (days).
    d_min : float
        Minimum luminosity distance in Mpc.
    d_max : float
        Maximum luminosity distance in Mpc.
    rate_density : float
        Volumetric GRB rate in events/Mpc^3/yr.

    Returns
    -------
    int
        Expected number of GRBs in the survey.
    """
    years = (t_end - t_start) / 365.25
    z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
    z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)

    V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    return np.random.poisson(rate_density * V * years)

# --------------------------------------------
# Alternate
# --------------------------------------------
def inject_uniform_healpix(nside, n_events, seed=42):
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    pix = rng.choice(npix, size=n_events)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = np.degrees(0.5 * np.pi - theta)
    return ra, dec
    
# --------------------------------------------
# GRB population generator
# --------------------------------------------
def generateGRBPopSlicer(t_start=1, t_end=3652, seed=42,
                         d_min=10, d_max=1000, num_lightcurves=1000, gal_lat_cut=None, rate_density=1e-8,
                         load_from=None, save_to=None):
    """
    Generate a population of GRB afterglows with realistic extinction and sky distribution.

    Parameters
    ----------
    gal_lat_cut : float or None
        Optional Galactic latitude cut (e.g., 15 deg).
    load_from : str or None
        If set, load slice_points from this pickle file.
    save_to : str or None
        If set, save the slice_points to this pickle file.
    """
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded GRB population from {load_from}")
        return slicer

    rng = np.random.default_rng(seed)
    n_events = sample_grb_rate_from_volume(t_start, t_end, d_min, d_max, rate_density=rate_density)
    print(f"Simulated {n_events} GRB events using rate_density = {rate_density:.1e}")

    
    #ra, dec = uniform_sphere_degrees(n_events, seed=seed) #returns degrees
    nside = 64  # Or 128 if you want higher resolution
    ra, dec = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)

    #print(f"[CHECK] Dec range: {dec.min():.2f} to {dec.max():.2f} (expected ~[-90, 90])")

    dec = np.clip(dec, -89.9999, 89.9999)
    #dec_rad = np.radians(dec)
    
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0) #returns radians 
    #print(f"Print 10 = {ra[:10],dec[:10]}")
    #print(f" Value = {slicer.slice_points}")
    #slicer.slice_points['ra'] = ra
    #slicer.slice_points['dec'] = dec_rad  # Correct assignment

    plt.hist(slicer.slice_points['ra'], bins=50)
    plt.xlabel("RA [rad]")
    plt.title("Injected GRB Population – RA Distribution")
    plt.grid(True)
    plt.show()
    
    plt.hist(slicer.slice_points['dec'], bins=50)
    plt.xlabel("Dec [rad]")
    plt.title("Injected GRB Population – Dec Distribution")
    plt.grid(True)
    plt.show()


    distances = rng.uniform(d_min, d_max, n_events)
    peak_times = rng.uniform(t_start, t_end, n_events)
    file_indx = rng.integers(0, num_lightcurves, len(ra))

    #print(f"[DEBUG] dec sample before SkyCoord: {dec[:5]}")
    #print(f"[DEBUG] dec units? min={np.min(dec):.2f}, max={np.max(dec):.2f}")
    print(f"[DEBUG]Print 5 sample before SkyCoord - ra,dec: {slicer.slice_points}")


    #coords = SkyCoord(ra=slicer.slice_points['ra'] * u.deg, dec=slicer.slice_points['dec'] * u.deg, frame='icrs') - this code just labels them as deg. u.deg doesn't convert them. 

    coords = SkyCoord(ra=np.degrees(slicer.slice_points['ra']) * u.deg, dec=np.degrees(slicer.slice_points['dec']) * u.deg, frame='icrs') #this line correctly converts them and labels them
    
    print(f"[DEBUG] coords.dec[:5]: {coords.dec[:5]}")
    print(f"[DEBUG] coords.dec.unit: {coords.dec.unit}")

    plt.hist(coords.ra, bins=50)
    plt.xlabel("RA [deg]")
    plt.title("SkyCoord RA Distribution")
    plt.grid(True)
    plt.show()
    
    plt.hist(coords.dec, bins=50)
    plt.xlabel("Dec [deg]")
    plt.title("SkyCoord Dec Distribution")
    plt.grid(True)
    plt.show()

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

    #slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    #slicer.slice_points['ra'] = ra
    #slicer.slice_points['dec'] = dec
    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    slicer.slice_points['gall'] = coords.galactic.l.deg
    slicer.slice_points['galb'] = coords.galactic.b.deg

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved GRB population to {save_to}")

    return slicer

    