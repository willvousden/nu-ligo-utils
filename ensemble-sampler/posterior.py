import numpy as np
import lal
import lalsimulation as ls
import utils as u

params_dtype = [('mc', np.float),
                ('eta', np.float),
                ('cos_iota', np.float),
                ('phi', np.float),
                ('psi', np.float),
                ('time', np.float),
                ('ra', np.float),
                ('sin_dec', np.float),
                ('log_dist', np.float)]

def to_params(arr):
    return arr.view(np.dtype(params_dtype))

def from_params(arr):
    return arr.view((np.float,9))

class Posterior(object):
    def __init__(self, time_data, srate=16384, time_offset=0,
                 approx=ls.TaylorF2, amp_order=-1, phase_order=-1, fmin=10.0):
        self._srate = srate
        self._data = np.fft.rfft(time_data)*(1.0/srate)
        self._T = srate*time_data.shape[0]
        self._fs = np.linspace(0, srate/2.0, self.data.shape[0])
        self._time_offset = time_offset
        self._approx = approx
        self._amp_order = amp_order
        self._phase_order = phase_order
        self._fmin = fmin

        # Throw away data below fmin
        sel = self.fs > fmin
        self._data = self._data[sel]
        self._fs = self._fs[sel]

        self._psd = np.zeros(self.fs.shape[0])
        for i in range(self.fs.shape[0]):
            self.psd[i] = ls.SimNoisePSDiLIGOModel(self.fs[i])
        
    @property
    def data(self):
        return self._data

    @property
    def T(self):
        return self._T

    @property
    def fs(self):
        return self._fs

    @property
    def srate(self):
        return self._srate
        
    @property
    def time_offset(self):
        return self._time_offset

    @property 
    def approx(self):
        return self._approx

    @property
    def amp_order(self):
        return self._amp_order

    @property
    def phase_order(self):
        return self._phase_order

    @property
    def fmin(self):
        return self._fmin

    @property
    def psd(self):
        return self._psd

    def generate_waveform(self, params):
        params = to_params(params)
        
        m1,m2 = u.mc_eta_to_masses(params['mc'], params['eta'])
        d = 1e6*lal.LAL_PC_SI*np.exp(params['log_dist'])
        i = np.arccos(params['cos_iota'])

        dec = np.arcsin(params['sin_dec'])[0]

        tsec,tfrac = np.modf(params['time'])

        tseconds = int(self.time_offset) + tsec
        tns = 1e9*tfrac

        tgps = lal.LIGOTimeGPS(0)

        tgps.gpsSeconds = int(tseconds)
        tgps.gpsNanoSeconds = int(tns)

        gmst = lal.GreenwichMeanSiderealTime(tgps)

        response = lal.lalCachedDetectors[lal.LALDetectorIndexLHODIFF].response
        location = lal.lalCachedDetectors[lal.LALDetectorIndexLHODIFF].location

        timedelay = lal.TimeDelayFromEarthCenter(location, params['ra'][0], dec, tgps)

        timeshift = int(self.time_offset) + params['time'] + timedelay

        fplus, fcross = lal.ComputeDetAMResponse(lal.lalCachedDetectors[lal.LALDetectorIndexLHODIFF].response,
                                                 params['ra'][0], dec, params['psi'][0], gmst)

        hplus,hcross = ls.SimInspiralChooseFDWaveform(params['phi'][0], 
                                                      self.fs[1]-self.fs[0],
                                                      m1[0]*lal.LAL_MSUN_SI, m2[0]*lal.LAL_MSUN_SI, 
                                                      0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0,
                                                      self.fs[0], 0.0,
                                                      d[0], i[0], 
                                                      0.0, 0.0,
                                                      None, None, 
                                                      self.amp_order, self.phase_order, 
                                                      self.approx, None)
        
        phase_from_timeshift = np.exp(-2.0j*np.pi*self.fs*timeshift)

        istart = int(np.round(self.fs[0]/(self.fs[1]-self.fs[0])))

        h = phase_from_timeshift[:hplus.data.length-istart]*(fplus*hplus.data.data[istart:] + fcross*hcross.data.data[istart:])

        return h
                                              
