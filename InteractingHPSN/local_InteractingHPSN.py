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

    import numpy as np
import os
import pickle
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from dustmaps.sfd import SFDQuery
from rubin_sim.maf.utils import m52snr
from rubin_sim.phot_utils import DustValues
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

# ---------------------------------------------------------------------
# Light Curve Parameters
# ---------------------------------------------------------------------
INTERACTING_HPSN_PARAMS = {
    'Icn': {
        'u': {
            'rise_rate_mu': 0.18,
            'rise_rate_sigma': 0.10,
            'fade_rate_mu': 0.16,
            'fade_rate_sigma': 0.05,
            'peak_mag_min': -20.0,
            'peak_mag_max': -18.5,
            'duration_at_peak': 4.0
        },
        'g': {
            'rise_rate_mu': 0.27,
            'rise_rate_sigma': 0.08,
            'fade_rate_mu': 0.14,
            'fade_rate_sigma': 0.06,
            'peak_mag_min': -20.0,
            'peak_mag_max': -17.0,
            'duration_at_peak': 4.6
        },
        'r': {
            'rise_rate_mu': 0.24,
            'rise_rate_sigma': 0.14,
            'fade_rate_mu': 0.20,
            'fade_rate_sigma': 0.10,
            'peak_mag_min': -19.5,
            'peak_mag_max': -17.0,
            'duration_at_peak': 3.7
        },
        'i': {
            'rise_rate_mu': 0.21,
            'rise_rate_sigma': 0.10,
            'fade_rate_mu': 0.14,
            'fade_rate_sigma': 0.05,
            'peak_mag_min': -19.0,
            'peak_mag_max': -17.0,
            'duration_at_peak': 5.0
        },
        'z': {
            'rise_rate_mu': 0.13,
            'rise_rate_sigma': 0.05,
            'fade_rate_mu': 0.15,
            'fade_rate_sigma': 0.10,
            'peak_mag_min': -19.0,
            'peak_mag_max': -18.0,
            'duration_at_peak': 5.0
        }
    },
    'Ibn': {
        'u': {
            'rise_rate_mu': 0.05,
            'rise_rate_sigma': 0.01,
            'fade_rate_mu': 0.12,
            'fade_rate_sigma': 0.02,
            'peak_mag_min': 18.29,
            'peak_mag_max': 18.29,
            'duration_at_peak': 3.4
        },
        'g': {
            'rise_rate_mu': 0.14,
            'rise_rate_sigma': 0.03,
            'fade_rate_mu': 0.12,
            'fade_rate_sigma': 0.01,
            'peak_mag_min': 17.18,
            'peak_mag_max': 18.84,
            'duration_at_peak': 2.3
        },
        'r': {
            'rise_rate_mu': 0.11,
            'rise_rate_sigma': 0.08,
            'fade_rate_mu': 0.13,
            'fade_rate_sigma': 0.01,
            'peak_mag_min': 14.36,
            'peak_mag_max': 18.95,
            'duration_at_peak': 3.17
        },
        'i': {
            'rise_rate_mu': 0.04,
            'rise_rate_sigma': 0.02,
            'fade_rate_mu': 0.12,
            'fade_rate_sigma': 0.03,
            'peak_mag_min': 14.37,
            'peak_mag_max': 19.14,
            'duration_at_peak': 3.0
        }
    }
}


# ---------------------------------------------------------------------
# Light Curve Generator for Interacting H-poor SNe
# ---------------------------------------------------------------------
class InteractingHPSN_LC:
    def __init__(self, num_samples=100, save_to=None, load_from=None):
        """
        Initialize the Interacting H-poor SN light curve generator.

        Parameters
        ----------
        num_samples : int
            Number of samples per phase segment.
        save_to : str or None
            If provided, saves generated templates to this pickle file.
        load_from : str or None
            If provided and file exists, loads templates instead of regenerating.
        """
        self.num_samples = num_samples
        self.filts = []
        self.templates = {'Ibn': [], 'Icn': []}

        if load_from and os.path.exists(load_from):
            with open(load_from, 'rb') as f:
                self.templates = pickle.load(f)
            print(f"Loaded InteractingHPSN light curves from {load_from}")
        else:
            for sn_type in self.templates:
                for _ in range(100):
                    self.templates[sn_type].append(self.generate_template(sn_type))
            if save_to:
                with open(save_to, 'wb') as f:
                    pickle.dump(self.templates, f)
                print(f"Saved 100 {sn_type} light curve templates to {save_to}")

    def generate_template(self, sn_type):
        """
        Generate a light curve template for a given SN type.

        Parameters
        ----------
        sn_type : str
            'Ibn' or 'Icn'.

        Returns
        -------
        dict
            Light curve dictionary for each band.
        """
        lc = {}
        self.filts = list(INTERACTING_HPSN_PARAMS[sn_type].keys())
        for band in self.filts:
            p = INTERACTING_HPSN_PARAMS[sn_type][band]

            rise_rate = np.random.normal(p['rise_rate_mu'], p['rise_rate_sigma'])
            fade_rate = np.random.normal(p['fade_rate_mu'], p['fade_rate_sigma'])
            peak_mag = np.random.uniform(p['peak_mag_min'], p['peak_mag_max'])
            peak_dur = p['duration_at_peak']

            t_rise = np.linspace(-5, 0, self.num_samples // 3)
            t_peak = np.linspace(0, peak_dur, self.num_samples // 3)
            t_fade = np.linspace(peak_dur, peak_dur + 10, self.num_samples // 3)

            mag_rise = peak_mag - rise_rate * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
            mag_peak = np.full_like(t_peak, peak_mag)
            mag_fade = peak_mag + fade_rate * (t_fade - t_fade[0])

            times = np.concatenate([t_rise, t_peak, t_fade])
            mags = np.concatenate([mag_rise, mag_peak, mag_fade])
            lc[band] = {'ph': times, 'mag': mags}
        return lc

    def interp(self, t, filtername, lc_template):
        return np.interp(t, lc_template[filtername]['ph'], lc_template[filtername]['mag'], left=99, right=99)


# ---------------------------------------------------------------------
# Base Metric
# ---------------------------------------------------------------------

class BaseInteractingHPSNMetric(BaseMetric):
    def __init__(self, metricName='BaseInteractingHPSNMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 outputLc=False, badval=-666, **kwargs):
        """
        Base class for Interacting H-poor SN metric evaluation logic.

        Parameters
        ----------
        outputLc : bool
            If True, returns matched light curves.
        """
        self.outputLc = outputLc
        self.ax1 = DustValues().ax1  # Extinction coefficient per filter

        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metricName=metricName,
                         units='Detected (0 or 1)', maps=['DustMap'],
                         badval=badval, **kwargs)

    def characterize_hpsn(self, snr, times):
        """
        Classify light curve as 'classical', 'ambiguous', or 'uncharacterized'.

        Returns
        -------
        str
            Classification string.
        """
        rise_window = (-5, 0)
        peak_window = (0, 5)
        fade_window = (5, 15)

        rise_pts = np.sum((times >= rise_window[0]) & (times < rise_window[1]) & (snr >= 0.5))
        peak_pts = np.sum((times >= peak_window[0]) & (times < peak_window[1]) & (snr >= 0.5))
        fade_pts = np.sum((times >= fade_window[0]) & (times < fade_window[1]) & (snr >= 0.5))

        total = rise_pts + peak_pts + fade_pts

        if rise_pts >= 2 and peak_pts >= 2 and fade_pts >= 1:
            return 'classical'
        elif total >= 5:
            return 'ambiguous'
        else:
            return 'uncharacterized'

    def evaluate(self, dataSlice, slicePoint):
        """
        Evaluate interpolated SN light curve against observation data.

        Returns
        -------
        dict
            Detection and classification information.
        """
        t = dataSlice['observationStartMJD'] - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)

        for i, f in enumerate(dataSlice['filter']):
            if f in slicePoint['lc']:
                mags[i] = np.interp(t[i], slicePoint['lc'][f]['ph'], slicePoint['lc'][f]['mag'])
                mags[i] += self.ax1.get(f, 0) * slicePoint['ebv']
            else:
                mags[i] = 99.0

        snr = m52snr(mags, dataSlice['fiveSigmaDepth'])
        classification = self.characterize_hpsn(snr, t)

        return {
            't': t,
            'snr': snr,
            'mags': mags,
            'classification': classification
        }

# ---------------------------------------------------------------------
# Metric Subclasses
# ---------------------------------------------------------------------
class InteractingHPSN_DetectMetric(BaseInteractingHPSNMetric):
    def run(self, dataSlice, slicePoint):
        result = self.evaluate(dataSlice, slicePoint)
        return int(np.sum(result['snr'] >= 5) >= 2)

class InteractingHPSN_ClassicalMetric(BaseInteractingHPSNMetric):
    def run(self, dataSlice, slicePoint):
        result = self.evaluate(dataSlice, slicePoint)
        return 1 if result['classification'] == 'classical' else 0

class InteractingHPSN_AmbiguousMetric(BaseInteractingHPSNMetric):
    def run(self, dataSlice, slicePoint):
        result = self.evaluate(dataSlice, slicePoint)
        return 1 if result['classification'] == 'ambiguous' else 0

class InteractingHPSN_UncharacterizedMetric(BaseInteractingHPSNMetric):
    def run(self, dataSlice, slicePoint):
        result = self.evaluate(dataSlice, slicePoint)
        return 1 if result['classification'] == 'uncharacterized' else 0

class InteractingHPSN_SeparabilityMetric(BaseInteractingHPSNMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(metricName='InteractingHPSN_Separability', *args, **kwargs)

    def run(self, dataSlice, slicePoint):
        result = self.evaluate(dataSlice, slicePoint)
        t = result['t']
        snr = result['snr']

        # Identify times when both g and r filters were observed
        g_inds = np.where(dataSlice['filter'] == 'g')[0]
        r_inds = np.where(dataSlice['filter'] == 'r')[0]

        if len(g_inds) < 2 or len(r_inds) < 2:
            return 0

        t_g = t[g_inds]
        t_r = t[r_inds]
        mags_g = result['mags'][g_inds]
        mags_r = result['mags'][r_inds]

        # Match closest times between g and r
        common_t = np.intersect1d(np.round(t_g, 1), np.round(t_r, 1))
        if len(common_t) < 2:
            return 0

        matched_g = [mags_g[np.argmin(np.abs(t_g - ct))] for ct in common_t]
        matched_r = [mags_r[np.argmin(np.abs(t_r - ct))] for ct in common_t]
        color = np.array(matched_g) - np.array(matched_r)

        try:
            slope = np.polyfit(common_t, color, 1)[0]
            return int(abs(slope) > 0.02)
        except:
            return 0

class InteractingHPSN_TriggerableMetric(BaseInteractingHPSNMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(metricName='InteractingHPSN_Triggerable', *args, **kwargs)

    def run(self, dataSlice, slicePoint):
        result = self.evaluate(dataSlice, slicePoint)
        t = result['t']
        snr = result['snr']

        # Trigger criteria: ≥2 filters with SNR ≥ 5 within 0.5 days
        filters = np.array(dataSlice['filter'])
        trigger_epochs = {}

        for i in range(len(t)):
            if snr[i] >= 5:
                epoch = np.floor(t[i] * 2) / 2  # Round to nearest 0.5 days
                if epoch not in trigger_epochs:
                    trigger_epochs[epoch] = set()
                trigger_epochs[epoch].add(filters[i])

        for filtset in trigger_epochs.values():
            if len(filtset) >= 2:
                return 1

        return 0


# ---------------------------------------------------------------------
# Population Generator for Interacting HPSN + automatic light curve injection
# ---------------------------------------------------------------------
def generateInteractingHPSNSlicer(prob_map, lc_generator, nside=64, t_start=1, t_end=3652,
                                  rate_per_year=6000, d_min=10, d_max=300,
                                  icn_fraction=0.05, gal_lat_cut=None, seed=42,
                                  save_to=None):
    from astropy.coordinates import SkyCoord
    from dustmaps.sfd import SFDQuery
    from rubin_sim.maf.slicers import UserPointsSlicer
    import astropy.units as u

    rng = np.random.default_rng(seed)
    n_years = (t_end - t_start) / 365.25
    n_events = int(rate_per_year * n_years)
    print(f"Generating {n_events} InteractingHPSN events")

    hp_indices = rng.choice(len(prob_map), size=n_events, p=prob_map)
    theta, phi = hp.pix2ang(nside, hp_indices, nest=False)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    if gal_lat_cut is not None:
        b = coords.galactic.b.deg
        mask = np.abs(b) > gal_lat_cut
        coords = coords[mask]

    distances = rng.uniform(d_min, d_max, len(coords))
    peak_times = rng.uniform(t_start, t_end, len(coords))
    subtypes = rng.choice(['Ibn', 'Icn'], p=[1 - icn_fraction, icn_fraction], size=len(coords))
    ebv = SFDQuery()(coords)

    slicer = UserPointsSlicer(ra=coords.ra.deg, dec=coords.dec.deg)
    slicer.slice_points['ra'] = coords.ra.deg
    slicer.slice_points['dec'] = coords.dec.deg
    slicer.slice_points['distance'] = distances
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['type'] = subtypes
    slicer.slice_points['ebv'] = ebv
    slicer.slice_points['sid'] = hp.ang2pix(nside, theta, phi, nest=False)
    slicer.slice_points['nside'] = nside

    # Attach light curves from generator
    attach_interacting_lcs_to_slicer(slicer, lc_generator)

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(dict(slicer.slice_points), f)
        print(f"Saved population to {save_to}")
    # Save the light curve templates used
    if save_to:
        template_file = os.path.splitext(save_to)[0] + "_templates.pkl"
        with open(template_file, 'wb') as f:
            pickle.dump(lc_generator.templates, f)
        print(f"Saved light curve templates to {template_file}")


    return slicer


