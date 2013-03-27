import numpy as np
import emcee
import lal
import lalsimulation as ls
import utils as u

nparams = 9
params_dtype = [('log_mc', np.float),
                ('eta', np.float),
                ('cos_iota', np.float),
                ('phi', np.float),
                ('psi', np.float),
                ('time', np.float),
                ('ra', np.float),
                ('sin_dec', np.float),
                ('log_dist', np.float)]

params_latex = [r'\log \mathcal{M}',
                r'\eta',
                r'\cos(\iota)',
                r'\phi',
                r'\psi',
                r't',
                r'\alpha',
                r'\sin(\delta)',
                r'\log d']

params_sigma = np.zeros(1, dtype=params_dtype)
params_sigma['log_mc'] = 1e-8
params_sigma['eta'] = 1e-6
params_sigma['ra'] = 1e-6
params_sigma['sin_dec'] = 1e-6
params_sigma['time'] = 1e-8
params_sigma['phi'] = 1e-3
params_sigma['psi'] = 1e-6
params_sigma['cos_iota'] = 1e-6
params_sigma['log_dist'] = 1e-4


def to_params(arr):
    """Returns a view of ``arr`` with labeled columns.  See
    :data:`params_dtype` for column names.
    """
    return arr.view(np.dtype(params_dtype))

def from_params(arr):
    """Returns a view of ``arr`` as a float array consistent with
    :data:`params_dtype`"""
    shape = arr.shape

    return arr.view(float).reshape(shape+(nparams,))

def sample_ball(params, size):
    """Returns an array of params of the given size in a small ball around
    the given parameters.
    """

    result = emcee.utils.sample_ball(params.view(float).reshape((-1,)),
                                     params_sigma.view(float).reshape((-1,)),
                                     size=size)

    result = result.view(params_dtype)
    
    for name,low,high in [('eta', 0.0, 0.25),
                          ('cos_iota', -1.0, 1.0),
                          ('phi', 0.0, 2.0*np.pi),
                          ('psi', 0.0, 2.0*np.pi),
                          ('ra', 0.0, 2.0*np.pi),
                          ('sin_dec', -1.0, 1.0)]:
        sel = result[name] < low
        result[name][sel] = 2.0*low - result[name][sel]

        sel = result[name] > high
        result[name][sel] = 2.0*high - result[name][sel]

    return result

class Posterior(object):
    """Callable object representing the posterior."""

    def __init__(self, time_data=None, inj_params=None, srate=16384,
                 T=None, time_offset=lal.LIGOTimeGPS(0),
                 approx=ls.TaylorF2, amp_order=-1, phase_order=-1,
                 fmin=40.0, malmquist_snr=None, mmin=1.0, mmax=35.0,
                 dmax=1000.0, dataseed=None, detectors=['H1', 'L1', 'V1']):
        r"""Set up the posterior.  Currently only does PE on H1 with iLIGO
        analytic noise spectrum.

        :param time_data: A list of float arrays giving the
          time-domain data in each detector on which the analysis is
          to operate.  If ``None``, then data are generated from
          Gaussian noise.

        :param inj_params: Parameters for the injected waveform.  If
          ``None``, no injection is performed.  

        :param srate: The sample rate, in Hz.

        :param T: The total length of the data segment (in seconds).
          If ``None``, extracted from ``time_data``.

        :param time_offset: The GPS start time of the segment being
          analyized.

        :param approx: The waveform approximant to use (currently only
          frequency-domain waveforms are supported).

        :param amp_order: The amplitude order parameter for the
          waveform.  Use ``-1`` for maximum order.

        :param phase_order: The phase order for the waveform.  Use
          ``-1`` for maximum order.

        :param fmin: The minimum frequency for the analysis.

        :param malmquist_snr: If not ``None``, gives the SNR threshold
          in the second-loudest detector (or only detector) below
          which the prior probability is zero.

        :param mmin: Minimum component mass threshold.

        :param mmax: Maximum component mass threshold.
        
        :param dmax: Maximum distance.

        :param dataseed: If not ``None``, will be used as a RNG seed
          for generating any synthetic data.

        :param detectors: The detectors on which the analysis runs.
        """

        self._srate = srate

        if T is None:
            self._T = time_data[0].shape[0]/srate
        else:
            self._T = T
            if time_data is not None:
                assert np.abs((T - time_data[0].shape[0]/srate)/T) < 1e-8, 'T does not match time_data shape'

        data_length = int(round(self.T*self.srate/2+1))

        self._fs = np.linspace(0, srate/2.0, self.T*self.srate/2+1)
        self._psd = np.zeros(self.fs.shape[0])
        for i in range(self.fs.shape[0]):
            self.psd[i] = ls.SimNoisePSDiLIGOModel(self.fs[i])

        if time_data is None:
            self._data = [np.zeros(data_length, dtype=np.complex) for d in detectors]

            # Maybe set seed?
            if dataseed is not None:
                old_state = np.random.get_state()
                np.random.seed(dataseed)
            
            for i in range(data_length):
                sigma = np.sqrt(self.psd[i])
                for j in range(len(detectors)):
                    self.data[j][i] = np.random.normal(loc=0.0, scale=sigma) + \
                                      1j*np.random.normal(loc=0.0, scale=sigma)

            # Reset random state
            if dataseed is not None:
                np.random.set_state(old_state)
        else:
            self._data = []
            for i in range(len(detectors)):
                self._data.append(np.fft.rfft(time_data[i])*(1.0/srate))
            assert data_length == self.data[0].shape[0], 'data_length and data.shape mismatch'

        self._time_offset = u.GPSTime(time_offset.gpsSeconds, time_offset.gpsNanoSeconds)
        self._approx = approx
        self._amp_order = amp_order
        self._phase_order = phase_order
        self._fmin = fmin
        self._malmquist_snr = malmquist_snr
        self._mmin = mmin
        self._mmax = mmax
        self._dmax = dmax
        self._detectors = detectors

        if inj_params is not None:
            hs = self.generate_waveform(inj_params)
            for i,h in enumerate(hs):
                self.data[i] += h
        
    @property
    def data(self):
        """The frequency-domain data on which the analysis will be conducted."""
        return self._data

    @property
    def T(self):
        """The length (in seconds) of the input data segment."""
        return self._T

    @property
    def fs(self):
        """The frequencies (in Hz) that correspond to the frequency domain data."""
        return self._fs

    @property
    def srate(self):
        """The sample rate of the time-domain data."""
        return self._srate
        
    @property
    def time_offset(self):
        """The GPS time of the start of the data segment."""
        return self._time_offset

    @property 
    def approx(self):
        """The waveform approximant."""
        return self._approx

    @property
    def amp_order(self):
        """Amplitude order (``-1`` for max order)."""
        return self._amp_order

    @property
    def phase_order(self):
        """The phase order (``-1`` for max order)."""
        return self._phase_order

    @property
    def fmin(self):
        """The minimum frequency of the analysis."""
        return self._fmin

    @property
    def psd(self):
        """The (one-sided) noise PSD used in the analysis."""
        return self._psd

    @property
    def malmquist_snr(self):
        """The SNR below which the prior goes to zero (or ``None`` for no threshold)."""
        return self._malmquist_snr

    @property
    def mmin(self):
        """The minimum component mass."""
        return self._mmin

    @property
    def mmax(self):
        """The maximum component mass."""
        return self._mmax

    @property
    def dmax(self):
        """The maximum distance."""
        return self._dmax

    @property
    def detectors(self):
        return self._detectors

    def generate_waveform(self, params):
        """Returns a frequency-domain strain suitable to subtract from the
        frequency-domain data (i.e. the samples line up in frequency
        space).
        """

        params = to_params(params)

        # Can only handle one parameter set at a time, so extract
        # first from array if more than one.
        if isinstance(params, np.ndarray):
            params = params[0]
        
        m1,m2 = u.mc_eta_to_masses(np.exp(params['log_mc']), params['eta'])
        d = 1e6*lal.LAL_PC_SI*np.exp(params['log_dist'])
        i = np.arccos(params['cos_iota'])

        dec = np.arcsin(params['sin_dec'])

        hplus,hcross = ls.SimInspiralChooseFDWaveform(params['phi'], 
                                                      self.fs[1]-self.fs[0],
                                                      m1*lal.LAL_MSUN_SI, m2*lal.LAL_MSUN_SI, 
                                                      0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0,
                                                      self.fmin, 0.0,
                                                      d, i, 
                                                      0.0, 0.0,
                                                      None, None, 
                                                      self.amp_order, self.phase_order, 
                                                      self.approx, None)

        hpdata = hplus.data.data
        hcdata = hcross.data.data

        # If necessary, cut down to size
        if hpdata.shape[0] > self.fs.shape[0]:
            hpdata = hpdata[:self.fs.shape[0]]
            hcdata = hcdata[:self.fs.shape[0]]

        N = hpdata.shape[0]

        hout=[]
        for d in self.detectors:
            tgps = lal.LIGOTimeGPS(0)
            tgps.gpsSeconds = self.time_offset.sec
            tgps.gpsNanoSeconds = self.time_offset.ns

            lal.GPSAddGPS(tgps, params['time'])

            gmst = lal.GreenwichMeanSiderealTime(tgps)

            if d == 'H1':
                diff = lal.LALDetectorIndexLHODIFF
            elif d == 'L1':
                diff = lal.LALDetectorIndexLLODIFF
            elif d == 'V1':
                diff = lal.LALDetectorIndexVIRGODIFF
            else:
                raise ValueError('detector not recognized: ' + d)
            
            response = lal.lalCachedDetectors[diff].response
            location = lal.lalCachedDetectors[diff].location

            timedelay = lal.TimeDelayFromEarthCenter(location, params['ra'], dec, tgps)

            timeshift = params['time'] + timedelay
                
            fplus, fcross = lal.ComputeDetAMResponse(lal.lalCachedDetectors[diff].response,
                                                     params['ra'], dec, params['psi'], gmst)

            phase_from_timeshift = np.exp(-2.0j*np.pi*self.fs[:N]*timeshift)

            h = phase_from_timeshift*(fplus*hpdata + fcross*hcdata)

            h_full = np.zeros(self.fs.shape[0], dtype=np.complex)
            h_full[:N] = h

            hout.append(h_full)

        return hout

    def log_likelihood(self, params):
        r"""Returns the log likelihood of the given parameters.  The
log-likelihood is

        .. math::
          
          \log \mathcal{L} = -\frac{1}{2} \left( -2 \Re \left\langle d | h \right\rangle + \left\langle h | h \right\rangle \right)

        where 

        .. math::

          \left\langle a | b \right\rangle = 4 \int df \, \frac{a^*(f) b(f)}{S(f)},

        where :math:`S(f)` is the one-sided noise power spectral density.

        This corresponds to the ususal log-likelihood in Gaussian
        noise, but normalized with respect to the likelihood for pure
        noise under the Gaussian assumption.  In other words, the
        log-likelihood with a zero signal is zero.

        If the :attr:`Posterior.malmquest_snr` is not ``None``, then
        the likelihood will be returned as ``float('-inf')`` when
        :math:`\left\langle h | h \right\rangle^{1/2}` is smaller than
        :attr:`Posterior.malmquest_snr`"""

        hs = self.generate_waveform(params)
        df = self.fs[1] - self.fs[0]

        istart = np.nonzero(self.fs >= self.fmin)[0][0]

        hh_list=[]
        logl = 0.0
        for h, d in zip(hs, self.data):
            hh = 4.0*df*np.real(np.conjugate(h)*h)/self.psd
            dh = 4.0*df*np.real(np.conjugate(d)*h)/self.psd

            hh = np.sum(hh[istart:])
            dh = np.sum(dh[istart:])

            hh_list.append(hh)

            logl += -0.5*(hh - 2.0*dh)

        # If malmquist priors, then cutoff when the SNR is too quiet.
        hh_list.sort()
        if self.malmquist_snr is not None:
            if len(hh) > 1 and hh[1] < self.malmquist_snr*self.malmquist_snr:
                return float('-inf')
            elif len(hh) == 1 and hh[0] < self.malmquist_snr*self.malmquist_snr:
                return float('-inf')

        return logl

    def log_prior(self, params):
        """Returns the log of the prior.  More details to follow.        
        """
        params = to_params(params)

        if isinstance(params, np.ndarray):
            params = params[0]

        if params['eta'] < 0 or params['eta'] > 0.25:
            return float('-inf')

        # First basic ranges on parameters:
        mc = np.exp(params['log_mc'])
        d = np.exp(params['log_dist'])
        m1,m2=u.mc_eta_to_masses(mc, params['eta'])
        mtot = m1+m2

        if m1 > self.mmax or m2 < self.mmin:
            return float('-inf')

        if params['cos_iota'] < -1.0 or params['cos_iota'] > 1.0:
            return float('-inf')

        if params['phi'] > 2.0*np.pi or params['phi'] < 0.0:
            return float('-inf')

        if params['psi'] > 2.0*np.pi or params['psi'] < 0.0:
            return float('-inf')

        if params['time'] < 0.0 or params['time'] > self.T:
            return float('-inf')

        if params['ra'] < 0.0 or params['ra'] > 2.0*np.pi:
            return float('-inf')

        if params['sin_dec'] < -1.0 or params['sin_dec'] > 1.0:
            return float('-inf')

        if d > self.dmax:
            return float('-inf')

        logp = 0.0

        # A flat prior in mass space gives the following in log(mc)-eta space:
        logp -= np.log(m1-m2) - 3.0*np.log(mtot)

        logp += 3.0*params['log_dist']

        return logp

    def __call__(self, params):
        lp = self.log_prior(params)

        if lp == float('-inf'):
            return lp

        return lp + self.log_likelihood(params)
