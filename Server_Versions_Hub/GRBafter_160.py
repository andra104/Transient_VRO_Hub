class GRB_lc:
    def __init__(self, num_samples=100):
        """
        GRBafterglows detection metric based on Igor Andreoni's kneMetrics.py file. 
        
        Generate synthetic GRB light curves using provided rise/fade rates.
        """
        self.data = []
        filts = ["u", "g", "r", "i", "z", "y"]
        
        # Define GRB light curve parameters
        rise_rates = {f: (6, 160) for f in filts}
        fade_rates = {f: (0.1, 10) for f in filts}
        peak_mag_range = {f: (-24, -22) for f in filts}
        duration_at_peak = {f: 0.1 for f in filts}
        
        # Define time grid (from peak to 30 days post-burst)
        t_rise = np.linspace(-1, 0, num_samples // 5)  # Rising phase
        t_fade = np.linspace(0.1, 30, num_samples)      # Fading phase
        
        for _ in range(100):  # Generate 100 synthetic light curves
            new_dict = {}
            for filt in filts:
                peak_mag = np.random.uniform(*peak_mag_range[filt])
                rise_rate = np.random.uniform(*rise_rates[filt])
                fade_rate = np.random.uniform(*fade_rates[filt])
                
                # Rising phase: Exponential brightening
                mag_rise = peak_mag - rise_rate * (t_rise - t_rise.min())
                
                # Peak duration
                mag_peak = np.full((1,), peak_mag)  # Convert scalar to 1D array
                
                # Fading phase: Power-law decay
                mag_fade = peak_mag + fade_rate * np.log10(t_fade)
                
                t_full = np.concatenate([t_rise, [0], t_fade])
                mag_full = np.concatenate([mag_rise, mag_peak, mag_fade])
                
                new_dict[filt] = {'ph': t_full, 'mag': mag_full}
            
            self.data.append(new_dict)
        
        print(f"Generated {len(self.data)} synthetic GRB light curves.")