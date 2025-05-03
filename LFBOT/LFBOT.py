from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from rubin_sim.utils import uniformSphere
from rubin_sim.photUtils import Dust_values
from rubin_sim.maf.utils import m52snr
import numpy as np
import matplotlib.pyplot as plt


class LFBOT_lc:
    """
    LFBOT detection metric based on Igor Andreoni's kneMetrics.py file. 

    Generates synthetic LFBOT light curves using rise/fade rates and peak magnitude constraints.
    """
    def __init__(self, num_samples=100):
        self.data = []
        filts = ["g", "r"]  # LFBOTs are detected in g and r bands

        # Light curve parameters for LFBOTs
        rise_rates = {'g': (0.25, 2.5), 'r': (0.25, 2.5)}
        fade_rates = {'g': (0.15, 0.45), 'r': (0.15, 0.45)}
        peak_mag_range = {'g': (-21.5, -20), 'r': (-21.5, -20)}
        duration_at_peak = {'g': 4, 'r': 4}  # Peak duration < 4 days

        t_rise = np.linspace(-4, 0, num_samples // 5)  # Rising phase
        t_fade = np.linspace(0.1, 30, num_samples)  # Fading phase

        def sample_rate(rate_range):
            """Sample a rate from a truncated normal distribution."""
            mu, sigma = np.mean(rate_range), (rate_range[1] - rate_range[0]) / 3
            return np.clip(np.random.normal(mu, sigma), rate_range[0], rate_range[1])

        for _ in range(100):  # Generate 100 synthetic light curves
            new_dict = {}
            for filt in filts:
                peak_mag = np.random.uniform(*peak_mag_range[filt])
                rise_rate = sample_rate(rise_rates[filt])
                fade_rate = sample_rate(fade_rates[filt])

                # Rising phase
                mag_rise = peak_mag - rise_rate * (t_rise - np.min(t_rise)) / np.ptp(t_rise)

                # Peak duration
                mag_peak = np.full((1,), peak_mag)

                # Fading phase
                mag_fade = peak_mag + fade_rate * (np.log10(1 + t_fade))

                t_full = np.concatenate([t_rise, [0], t_fade])
                mag_full = np.concatenate([mag_rise, mag_peak, mag_fade])

                new_dict[filt] = {'ph': t_full, 'mag': mag_full}

            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        return np.interp(t, self.data[lc_indx][filtername]['ph'],
                         self.data[lc_indx][filtername]['mag'],
                         left=99, right=99)
    
    def plot_light_curves(self, num_to_plot=10):
        """
        Plot a subset of generated LFBOT light curves for g and r filters.
        Each filter will have its own plot.
        
        Parameters:
        num_to_plot : int
            Number of light curves to plot per filter.
        """
        filters = ["g", "r"]
        
        for filter_name in filters:
            plt.figure(figsize=(8, 6))
            
            for i in range(min(num_to_plot, len(self.data))):
                time = self.data[i][filter_name]['ph']
                mag = self.data[i][filter_name]['mag']
                plt.plot(time, mag, label=f'LFBOT {i+1}', alpha=0.7)

            plt.gca().invert_yaxis()  # Magnitudes: lower values are brighter
            plt.xlabel("Time (days)")
            plt.ylabel("Magnitude")
            plt.title(f"LFBOT Light Curves in {filter_name}-band")
            plt.legend(loc='best', fontsize=8)
            plt.grid(True)
            plt.show()


class LFBOTPopMetric(BaseMetric):
    def __init__(self, metricName='LFBOTPopMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', nightCol='night',
                 ptsNeeded=2, mjd0=59853.5, outputLc=False, badval=-666, **kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.outputLc = outputLc
        self.lightcurves = LFBOT_lc(num_samples=100)
        self.mjd0 = mjd0
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1
        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]

        super().__init__(col=cols, units='Detected, 0 or 1',
                         metricName=metricName, maps=maps, badval=badval, **kwargs)

    def process_data(self, dataSlice, slicePoint):
        """
        Handles magnitude interpolation and extinction corrections for LFBOT light curves.
        """
        t = dataSlice[self.mjdCol] - self.mjd0 - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)
    
        for filtername in np.unique(dataSlice[self.filterCol]):
            if filtername not in ['g', 'r']:  # Ensure we only process valid filters
                continue
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



class LFBOTPopMetricDetect(LFBOTPopMetric):
    def run(self, dataSlice, slicePoint=None):
        around_peak, filters, times = self.process_data(dataSlice, slicePoint)

        if len(times) < 2:  # Need at least 2 detections
            return float(0)

        times = np.sort(times)  # Ensure chronological order

        # Compute time differences between detections
        time_diffs = np.diff(times)

        # Condition 1: Remove moving objects (require separation > 30 min)
        if np.any(time_diffs < (30 / 1440)):  # 30 minutes = 30/1440 days
            return float(0)

        # Condition 2: Ensure fast evolution (detections within 6 days)
        if (times[-1] - times[0]) > 6:  # Max separation between detections
            return float(0)

        # Condition 3: Require at least one epoch with detections in ≥2 filters
        if len(np.unique(filters)) >= 2:
            return float(1)

        # Condition 4: If detected post-peak, require 3 epochs, duration under 9 days
        if len(filters) >= 3 and (times[-1] - times[0]) <= 9:
            return float(1)

        return float(0)



def generateLFBOTPopSlicer(t_start=1, t_end=3652, seed=42, n_files=100, d_min=10, d_max=600):
    """
    Generate a population slicer for LFBOT events with a specified distance range.
    """
    np.random.seed(seed)
    n_events = np.random.poisson(0.3 * 420e-9 * 5e9 * (t_end - t_start) / 365.25)  # Adjust rate density

    ra, dec = uniformSphere(n_events, seed=seed)
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.random.randint(0, n_files, size=n_events)
    distances = np.random.uniform(d_min, d_max, size=n_events)

    slicer = UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    slicer.slicePoints['peak_time'] = peak_times
    slicer.slicePoints['file_indx'] = file_indx
    slicer.slicePoints['distance'] = distances  

    return slicer
