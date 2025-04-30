import numpy as np
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from rubin_sim.utils import uniformSphere
from rubin_sim.maf.utils import m52snr
from rubin_sim.utils import galacticFromEquatorial
import matplotlib.pyplot as plt


__all__ = ['MDwarfFlareMetric', 'generateMDwarfFlareSlicer']

class MDwarfFlareMetric(BaseMetric):
    """
    MDwarfFlares detection metric based on Igor Andreoni's kneMetrics.py file. 
    Generate synthetic GRB light curves using provided rise/fade rates.
    """
    def __init__(self, metricName='MDwarfFlareMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', nightCol='night',
                 ptsNeeded=2, outputLc=False, badval=-666, **kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.outputLc = outputLc
        
        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super().__init__(col=cols, units='Detected, 0 or 1',
                         metricName=metricName, maps=maps, badval=badval,
                         **kwargs)
    
    def _multi_detect(self, around_peak):
        if np.size(around_peak) < 3:
            return 0
        if np.any(around_peak >= 5):
            return 1
        return 0
    
    def _epoch_detect(self, t, mags, mags_unc, filters):
        if len(t) < 2:
            return 0
        dt = np.diff(t)
        if np.any(dt > 0.5):
            return 0
        if np.any(dt > 1/96):  # 15 minutes in days
            return 1
        if np.max(t) - np.min(t) > 1:
            return 0
        return 1
    
    def _flare_complexity(self, mags, threshold=1.5):
        peaks = np.where(mags > threshold)[0]
        return 1 if len(peaks) >= 2 else 0

    def reduce_multi_detect(self, metric):
        if isinstance(metric, dict):
            metric = metric.get('multi_detect', 0.0)  # Extract correct value
    
        if not isinstance(metric, (int, float)):  
            return 0.0  # Ensure it returns a float
    
        if not hasattr(self, "debug_reduce_counter_multi"):
            self.debug_reduce_counter_multi = 0
        if self.debug_reduce_counter_multi < 5:
            with open("debug_log.txt", "a") as log_file:
                log_file.write(f"DEBUG: Running reduce_multi_detect on {metric}\n")
            self.debug_reduce_counter_multi += 1
        return float(metric)

   
        
    def reduce_epoch_detect(self, metric):
        if isinstance(metric, dict):
            metric = metric.get('epoch_detect', 0.0)  # Extract correct value
    
        if not isinstance(metric, (int, float)):  
            return 0.0  # Ensure it returns a float
    
        if not hasattr(self, "debug_reduce_counter_epoch"):
            self.debug_reduce_counter_epoch = 0
        if self.debug_reduce_counter_epoch < 5:
            with open("debug_log.txt", "a") as log_file:
                log_file.write(f"DEBUG: Running reduce_epoch_detect on {metric}\n")
            self.debug_reduce_counter_epoch += 1
        return float(metric)


    def reduce_flare_complexity(self, metric):
        if isinstance(metric, dict):
            metric = metric.get('flare_complexity', 0.0)  # Extract correct value
    
        if not isinstance(metric, (int, float)):  
            return 0.0  # Ensure it returns a float
    
        if not hasattr(self, "debug_reduce_counter_flare"):
            self.debug_reduce_counter_flare = 0
        if self.debug_reduce_counter_flare < 5:
            with open("debug_log.txt", "a") as log_file:
                log_file.write(f"DEBUG: Running reduce_flare_complexity on {metric}\n")
            self.debug_reduce_counter_flare += 1
        return float(metric)



    def run(self, dataSlice, slicePoint=None):
        # Step 2: Prevent redundant execution
        if hasattr(self, "metricValues") and self.metricValues is not None:
            print(f"DEBUG: Skipping redundant execution for slice {slicePoint['sid']}")
            return self.metricValues
    
        result = {}
        t = dataSlice[self.mjdCol] - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)
    
        for filtername in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == filtername)
            mags[infilt] = np.random.uniform(12, 16, size=len(infilt[0]))  # Simulated mag values
    
        # Step 1: Track unique slice IDs
        if not hasattr(self, "processed_slices"):
            self.processed_slices = set()
    
        if slicePoint is not None and slicePoint['sid'] in self.processed_slices:
            print(f"DEBUG: Skipping slice {slicePoint['sid']} as it has already been processed.")
            return None  # Prevent redundant execution
        
        self.processed_slices.add(slicePoint['sid'])  # Add to processed set
    
        # Compute detection metrics
        around_peak = np.where(mags < dataSlice[self.m5Col] - 5)[0]
        filters = dataSlice[self.filterCol][around_peak]
    
        snr = m52snr(mags, dataSlice[self.m5Col])
        mags_unc = 2.5 * np.log10(1 + 1./snr)
    
        result['multi_detect'] = self._multi_detect(around_peak)
        result['epoch_detect'] = self._epoch_detect(t, mags, mags_unc, filters)
        result['flare_complexity'] = float(self._flare_complexity(mags))
    
        # Step 3: Debugging - Check metric values before returning
        if not hasattr(self, "debug_counter"):
            self.debug_counter = 0
        if self.debug_counter < 5:
            print(f"DEBUG [{self.debug_counter}]: multi_detect type = {type(result['multi_detect'])}")
            print(f"DEBUG [{self.debug_counter}]: epoch_detect type = {type(result['epoch_detect'])}")
            print(f"DEBUG [{self.debug_counter}]: flare_complexity type = {type(result['flare_complexity'])}")
            self.debug_counter += 1
    
        return result
        
def generateMDwarfFlareSlicer(t_start=1, t_end=3652, seed=42):
    """ Generate M Dwarf Flare events with a latitude-dependent rate. """

    # Full sky coverage
    total_sky_area = 41253  # Full sky in deg²

    # M Dwarf flare occurrence rate (average of 0.7 - 2.0 deg⁻² hr⁻¹)
    avg_occurrence_rate = 1.35  # Flares per deg² per hour

    # Compute expected total flares based on occurrence rate
    hours_per_day = 24
    total_days = t_end - t_start
    total_flares = avg_occurrence_rate * total_sky_area * hours_per_day * total_days

    # Apply a cap only if necessary
    max_events = 100000  # Prevents memory overload
    n_events = min(int(total_flares), max_events)

    print(f"Simulating {n_events} M Dwarf Flares with latitude-dependent weighting.")

    # Generate uniform RA/Dec positions
    ra, dec = uniformSphere(n_events, seed=seed)

    # Convert to Galactic coordinates
    l, b = galacticFromEquatorial(ra, dec)

    # Apply latitude-dependent weighting: fewer flares at high |b|
    weight = np.exp(-np.abs(b) / 20.0)  # Exponential decay with latitude
    keep_prob = weight / weight.max()  # Normalize to max 1.0

    # Randomly filter flares based on keep_prob
    mask = np.random.uniform(0, 1, n_events) < keep_prob
    ra, dec = ra[mask], dec[mask]

    # Update n_events to reflect the filtered sample
    n_events = len(ra)

    # Assign peak times
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)

    # Create the slicer with the updated flare distribution
    slicer = UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    slicer.slicePoints['peak_time'] = peak_times

    print(f"Final sample: {n_events} flares (after latitude weighting)")

    return slicer
