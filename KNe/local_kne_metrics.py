from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix

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
    'KN_lc', 'KNePopMetric', 'generateKNPopSlicer',
    'KNeDetectMetric', 'KNeZTFRestSimpleMetric', 'KNeZTFRestSimpleRedMetric',
    'KNeZTFRestSimpleBlueMetric', 'KNeMultiColorDetectMetric',
    'KNeRedColorDetectMetric', 'KNeBlueColorDetectMetric'
]



from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix

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
    'KN_lc', 'KNePopMetric', 'generateKNPopSlicer',
    'KNeDetectMetric', 'KNeZTFRestSimpleMetric', 'KNeZTFRestSimpleRedMetric',
    'KNeZTFRestSimpleBlueMetric', 'KNeMultiColorDetectMetric',
    'KNeRedColorDetectMetric', 'KNeBlueColorDetectMetric'
]


def get_filename(inj_params_list):
    """Given kilonova parameters, get the filename from the grid of models
    developed by M. Bulla

    Parameters
    ----------
    inj_params_list : list of dict
        parameters for the kilonova model such as
        mass of the dynamical ejecta (mej_dyn), mass of the disk wind ejecta
        (mej_wind), semi opening angle of the cylindrically-symmetric ejecta
        fan ('phi'), and viewing angle ('theta'). For example
        inj_params_list = [{'mej_dyn': 0.005,
              'mej_wind': 0.050,
              'phi': 30,
              'theta': 25.8}]
    """
    # Get files, model grid developed by M. Bulla
    datadir = get_data_dir()
    file_list = glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat'))
 
    params = {}
    matched_files = []
    for filename in file_list:
        key = filename.replace(".dat","").split("/")[-1]
        params[key] = {}
        params[key]["filename"] = filename
        keySplit = key.split("_")
        # Binary neutron star merger models
        if keySplit[0] == "nsns":
            mejdyn = float(keySplit[2].replace("mejdyn",""))
            mejwind = float(keySplit[3].replace("mejwind",""))
            phi0 = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])
            params[key]["mej_dyn"] = mejdyn
            params[key]["mej_wind"] = mejwind
            params[key]["phi"] = phi0
            params[key]["theta"] = theta
        # Neutron star--black hole merger models
        elif keySplit[0] == "nsbh":
            mej_dyn = float(keySplit[2].replace("mejdyn",""))
            mej_wind = float(keySplit[3].replace("mejwind",""))
            phi = float(keySplit[4].replace("phi",""))
            theta = float(keySplit[5])
            params[key]["mej_dyn"] = mej_dyn
            params[key]["mej_wind"] = mej_wind
            params[key]["phi"] = phi
            params[key]["theta"] = theta
    for key in params.keys():
        for inj_params in inj_params_list:
            match = all([np.isclose(params[key][var],inj_params[var]) for var in inj_params.keys()])
            if match:
                matched_files.append(params[key]["filename"])
                print(f"Found match for {inj_params}")
    print(f"Found matches for {len(matched_files)}/{len(inj_params_list)} \
          sets of parameters")

    return matched_files




class KN_lc:
    def __init__(self, file_list=None):
        if file_list is None:
            datadir = get_data_dir()
            file_list = glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat'))
        filts = ["u", "g", "r", "i", "z", "y"]
        magidxs = [1, 2, 3, 4, 5, 6]
        self.data = []
        for filename in file_list:
            mag_ds = np.loadtxt(filename)
            t = mag_ds[:, 0]
            new_dict = {}
            for filt, magidx in zip(filts, magidxs):
                new_dict[filt] = {'ph': t, 'mag': mag_ds[:, magidx]}
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)


class KNePopMetric(BaseMetric):
    def __init__(self, metricName='KNePopMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', nightCol='night',
                 ptsNeeded=2, file_list=None, mjd0=59853.5, badval=-666, **kwargs):
        self.metric_name = metricName
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.mjd0 = mjd0
        self.lightcurves = KN_lc(file_list=file_list)
        self.ax1 = DustValues().ax1

        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super().__init__(col=cols, metric_name=metricName, units='Detected, 0 or 1',
                         maps=['DustMap'], badval=badval, **kwargs)

    def _compute_mags(self, dataSlice, t, slice_point):
        mags = np.zeros(t.size, dtype=float)
        for f in np.unique(dataSlice[self.filterCol]):
            idx = np.where(dataSlice[self.filterCol] == f)
            mags[idx] = self.lightcurves.interp(t[idx], f, lc_indx=slice_point['file_indx'])
            mags[idx] += self.ax1[f] * slice_point['ebv']
            mags[idx] += 5 * np.log10(slice_point['distance'] * 1e6) - 5
            return mags

    def _ztfrest_filter_logic(self, around_peak, mags, mags_unc, t, filters, allowed_filters):
        if len(around_peak) < self.ptsNeeded:
            return 0
        if np.max(t[around_peak]) - np.min(t[around_peak]) < 0.125:
            return 0

        for f in allowed_filters:
            f_idx = np.where(filters == f)[0]
            if len(f_idx) < 2:
                continue
            times_f = t[around_peak][f_idx]
            mags_f = mags[around_peak][f_idx]
            mags_unc_f = mags_unc[around_peak][f_idx]
            dt = np.abs(times_f[np.argmax(mags_f)] - times_f[np.argmin(mags_f)])
            if dt < 0.125:
                continue
            brightening = mags_f[np.argmin(mags_f)] + mags_unc_f[np.argmin(mags_f)]
            fading = mags_f[np.argmax(mags_f)] - mags_unc_f[np.argmax(mags_f)]
            if brightening < fading:
                evol_rate = (np.max(mags_f) - np.min(mags_f)) / dt
                if evol_rate >= 0.3 or evol_rate <= -1.0:
                    return 1
        return 0

    def _multi_color_detect(self, filters):
        return int(len(np.unique(filters)) >= 2)

    def _red_color_detect(self, filters, min_det=4):
        return int(sum(np.isin(filters, ['i', 'z', 'y'])) >= min_det)

    def _blue_color_detect(self, filters, min_det=4):
        return int(sum(np.isin(filters, ['u', 'g', 'r'])) >= min_det)


class KNeDetectMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='multi_detect', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            return int(len(around_peak) >= self.ptsNeeded)
        except Exception:
            return self.badval


class KNeZTFRestSimpleMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='ztfrest_simple', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            snr = m52snr(mags, dataSlice[self.m5Col])
            mags_unc = 2.5 * np.log10(1. + 1. / snr)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            filters = dataSlice[self.filterCol][around_peak]
            return self._ztfrest_filter_logic(around_peak, mags, mags_unc, t, filters, allowed_filters='ugrizy')
        except Exception:
            return self.badval


class KNeZTFRestSimpleRedMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='ztfrest_simple_red', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            snr = m52snr(mags, dataSlice[self.m5Col])
            mags_unc = 2.5 * np.log10(1. + 1. / snr)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            filters = dataSlice[self.filterCol][around_peak]
            return self._ztfrest_filter_logic(around_peak, mags, mags_unc, t, filters, allowed_filters='izy')
        except Exception:
            return self.badval


class KNeZTFRestSimpleBlueMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='ztfrest_simple_blue', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            snr = m52snr(mags, dataSlice[self.m5Col])
            mags_unc = 2.5 * np.log10(1. + 1. / snr)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            filters = dataSlice[self.filterCol][around_peak]
            return self._ztfrest_filter_logic(around_peak, mags, mags_unc, t, filters, allowed_filters='ugr')
        except Exception:
            return self.badval


class KNeMultiColorDetectMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='multi_color_detect', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            filters = dataSlice[self.filterCol][around_peak]
            return self._multi_color_detect(filters)
        except Exception:
            return self.badval


class KNeRedColorDetectMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='red_color_detect', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            filters = dataSlice[self.filterCol][around_peak]
            return self._red_color_detect(filters)
        except Exception:
            return self.badval


class KNeBlueColorDetectMetric(KNePopMetric):
    def __init__(self, **kwargs):
        super().__init__(metricName='blue_color_detect', **kwargs)

    def run(self, dataSlice, slice_point=None):
        try:
            t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
            mags = self._compute_mags(dataSlice, t, slice_point)
            around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
            filters = dataSlice[self.filterCol][around_peak]
            return self._blue_color_detect(filters)
        except Exception:
            return self.badval


def generateKNPopSlicer(t_start=1, t_end=3652, n_events=10000, seed=42,
                        n_files=100, d_min=10, d_max=300, nside=64,
                        save_to="kne_templates_used.pkl"):
    def rndm(a, b, g, size=1):
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag) * r)**(1. / g)

    datadir = get_data_dir()
    file_list = sorted(glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat')))
    rng = np.random.default_rng(seed)
    selected_files = rng.choice(file_list, size=n_files, replace=False)

    if save_to:
        metadata = [{"filename": path} for path in selected_files]
        with open(save_to, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Saved selected KNe templates to {save_to}")

    ra, dec = inject_uniform_healpix(nside=nside, n_events=n_events, seed=seed)
    peak_times = rng.uniform(t_start, t_end, n_events)
    file_indx = rng.integers(0, n_files, size=n_events)
    distance = rndm(d_min, d_max, 4, size=n_events)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    ebv = SFDQuery()(coords)

    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    slicer.slice_points.update({
        'ra': np.radians(ra),
        'dec': np.radians(dec),
        'peak_time': peak_times,
        'file_indx': file_indx,
        'distance': distance,
        'ebv': ebv,
        'template_filenames': [selected_files[i] for i in file_indx]
    })
    return slicer





