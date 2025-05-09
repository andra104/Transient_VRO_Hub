    def evaluate_lc(self, dataSlice, slice_point, return_full_obs=True):
        t = dataSlice[self.mjdCol] - self.mjd0 - slice_point['peak_time']
    
        # ✅ Restrict to selected filters early
        if self.filter_include is not None:
            keep = np.isin(dataSlice[self.filterCol], self.filter_include)
            dataSlice = dataSlice[keep]
            t = t[keep]
    
        mags = np.zeros(t.size)
    
        for f in np.unique(dataSlice[self.filterCol]):
            if f not in self.lc_model.filts:
                continue
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
                'filter': filters
            }
            return snr, filters, times, obs_record
        return snr, filters, times
