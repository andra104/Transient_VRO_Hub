from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from rubin_sim.utils import uniformSphere
from rubin_sim.photUtils import Dust_values
from rubin_sim.data import get_data_dir
from rubin_sim.maf.utils import m52snr
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import glob
import os

class GRB_lc:
    def __init__(self, num_samples=100):
        """
        GRBafterglows detection metric based on Igor Andreoni's kneMetrics.py file. 
        
        Generate synthetic GRB light curves using provided rise/fade rates.
        """
        self.data = []
        filts = ["u", "g", "r", "i", "z", "y"]
        
        # Define realistic GRB light curve parameters
        rise_rates = {f: (0.5, 10) for f in filts}  # Realistic optical rise rate range
        fade_rates = {f: (0.1, 5) for f in filts}   # Realistic optical fade rate range
        peak_mag_range = {f: (-24, -22) for f in filts}  # Peak brightness range
        duration_at_peak = {f: 0.1 for f in filts}  # Time at peak magnitude
        
        # Time grid (from peak to 30 days post-burst)
        t_rise = np.linspace(-1, 0, num_samples // 5)  # Rising phase
        t_fade = np.linspace(0.1, 30, num_samples)      # Fading phase
        
        # Use normal distribution to sample rise/fade rates
        def sample_rate(rate_range):
            """Sample a rate from a truncated normal distribution."""
            mu, sigma = np.mean(rate_range), (rate_range[1] - rate_range[0]) / 3  # 3-sigma spread
            return np.clip(np.random.normal(mu, sigma), rate_range[0], rate_range[1])
        
        for _ in range(100):  # Generate 100 synthetic light curves
            new_dict = {}
            for filt in filts:
                peak_mag = np.random.uniform(*peak_mag_range[filt])
                rise_rate = sample_rate(rise_rates[filt])
                fade_rate = sample_rate(fade_rates[filt])
                
                # Rising phase: Exponential brightening
                mag_rise = peak_mag - rise_rate * (t_rise - np.min(t_rise)) / np.ptp(t_rise)
                
                # Peak duration
                mag_peak = np.full((1,), peak_mag)  # Convert scalar to 1D array
                
                # Fading phase: Log-based decay to prevent extreme drops
                mag_fade = peak_mag + fade_rate * (np.log10(1 + t_fade))
                
                t_full = np.concatenate([t_rise, [0], t_fade])
                mag_full = np.concatenate([mag_rise, mag_peak, mag_fade])
                
                new_dict[filt] = {'ph': t_full, 'mag': mag_full}
            
            self.data.append(new_dict)
        
        print(f"Generated {len(self.data)} synthetic GRB light curves.")
    
    def plot_light_curves(self, num_to_plot=10, filter_name="r"):
        """
        Plot a subset of generated GRB light curves in a given filter.
        """
        colors = plt.cm.viridis(np.linspace(0, 1, num_to_plot))
        plt.figure(figsize=(10, 6))
        
        for i in range(num_to_plot):
            time = self.data[i][filter_name]['ph']
            mag = self.data[i][filter_name]['mag']
            plt.plot(time, mag, label=f'GRB {i+1}', color=colors[i])
        
        plt.gca().invert_yaxis()  # Magnitudes: lower values are brighter
        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude")
        plt.title(f"Simulated GRB Light Curves ({filter_name}-band)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def interp(self, t, filtername, lc_indx=0):
        if lc_indx >= len(self.data):
            print(f"Warning: lc_indx {lc_indx} is out of range (max {len(self.data)-1}). Using the last available light curve.")
            lc_indx = len(self.data) - 1  # Use last available file instead
        
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)


class GRBPopMetric(BaseMetric):
    def __init__(self, metricName='GRBPopMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', nightCol='night',
                 ptsNeeded=2, file_list=None, mjd0=59853.5, outputLc=False, badval=-666,
                 **kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.outputLc = outputLc
        self.lightcurves = GRB_lc(num_samples=100)  # Generate synthetic GRB light curves
        self.mjd0 = mjd0
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1
        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]

        super(GRBPopMetric, self).__init__(col=cols, units='Detected, 0 or 1',
                                           metricName=metricName, maps=maps, badval=badval,
                                           **kwargs)

    def process_data(self, dataSlice, slicePoint):
        """
        Handles magnitude interpolation and extinction corrections.
        """
        t = dataSlice[self.mjdCol] - self.mjd0 - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)

        for filtername in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt], filtername,
                                                   lc_indx=slicePoint['file_indx'])
            A_x = self.Ax1[filtername] * slicePoint['ebv']
            mags[infilt] += A_x
            distmod = 5 * np.log10(slicePoint['distance'] * 1e6) - 5.0
            mags[infilt] += distmod

        around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
        filters = dataSlice[self.filterCol][around_peak]
        times = t[around_peak]

        return around_peak, filters, times


class GRBPopMetricMultiDetect(GRBPopMetric):
    def run(self, dataSlice, slicePoint=None):
        around_peak, filters, times = self.process_data(dataSlice, slicePoint)

        # Ensure at least 2 detections in the same filter
        for f in np.unique(filters):
            mask = filters == f
            if np.sum(mask) >= 2:
                time_diff = np.max(times[mask]) - np.min(times[mask])
                if time_diff >= 0.5 / 24:  # 30 minutes in days
                    return float(1)
        return float(0)


class GRBPopMetricEpochDetect(GRBPopMetric):
    def run(self, dataSlice, slicePoint=None):
        around_peak, filters, times = self.process_data(dataSlice, slicePoint)

        # First epoch: non-detection allowed
        # Second epoch: must have detection in 2 different filters
        if len(np.unique(filters)) >= 2:
            return float(1)
        return float(0)


def sample_grb_rate(t_start, t_end, rate_density=1e-7, volume=5e9):
    """
    Compute the expected number of GRB events based on astrophysical rates.

    Parameters:
    - t_start, t_end: Time range in days.
    - rate_density: GRB rate per Mpc³ per year (~10⁻⁹ is too low; try ~10⁻⁷).
    - volume: Comoving volume in Mpc³ (~5×10⁹ Mpc³ for LSST range).

    Returns:
    - n_events: Poisson-sampled number of GRBs.
    """
    years = (t_end - t_start) / 365.25
    expected_events = rate_density * volume * years
    return max(100, np.random.poisson(expected_events))  # Ensure minimum 100 events


def generateGRBPopSlicer(t_start=1, t_end=3652, seed=42, n_files=100, d_min=10, d_max=1000):
    """
    Generate a population slicer for GRB events with a specified distance range.
    """
    np.random.seed(seed)

    # Sample the number of events dynamically
    n_events = sample_grb_rate(t_start, t_end)

    # Sample sky positions
    ra, dec = uniformSphere(n_events, seed=seed)

    # Sample peak times
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)

    # Assign random file indices for light curves
    file_indx = np.random.randint(0, n_files, size=n_events)

    # Sample distances within the given range
    distances = np.random.uniform(d_min, d_max, size=n_events)

    # Create the slicer
    slicer = UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    slicer.slicePoints['peak_time'] = peak_times
    slicer.slicePoints['file_indx'] = file_indx
    slicer.slicePoints['distance'] = distances  

    return slicer

