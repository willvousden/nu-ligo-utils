import numpy as np
import emcee
import fftw3
import lal
import lalsimulation as ls
from params import to_params, params_dtype, params_to_time_marginalized_params, time_marginalized_params_to_params, to_time_marginalized_params
from posterior_utils import *
import scipy.special as ss
import utils as u

class Posterior(object):
    """Callable object representing the posterior."""

    def __init__(self, time_data=None, inj_params=None, srate=16384,
                 T=None, time_offset=lal.LIGOTimeGPS(0),
                 approx=ls.TaylorF2, amp_order=-1, phase_order=-1,
                 fmin=40.0, malmquist_snr=None, mmin=1.0, mmax=35.0,
                 dmax=1000.0, dataseed=None, detectors=['H1', 'L1',
                                                        'V1']):
        r"""Set up the posterior.  Currently only does PE on H1 with iIGOIGO
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
            self.psd[i] = ls.SimNoisePSDaLIGOZeroDetHighPower(self.fs[i])

        # Zero out signals below fmin:
        self.psd[self.fs < fmin] = float('inf')

        if time_data is None:
            self._data = [np.zeros(data_length, dtype=np.complex) for d in detectors]

            # Maybe set seed?
            if dataseed is not None:
                old_state = np.random.get_state()
                np.random.seed(dataseed)
            
            for j in range(len(detectors)):
                sigma = np.sqrt(self.psd)
                self.data[j] = sigma*(np.random.normal(size=data_length) + \
                                      np.random.normal(size=data_length)*1j)
                self.data[j][sigma==float('inf')] = 0.0

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
        self._msnr = malmquist_snr
        self._mmin = mmin
        self._mmax = mmax
        self._dmax = dmax
        self._detectors = detectors

        self._c2r_input_fft_array = np.zeros(self.data[0].shape[0], dtype=np.complex128)
        self._c2r_output_fft_array = np.zeros((self.data[0].shape[0]-1)*2, dtype=np.float64)
        self._c2r_fft_plan = fftw3.Plan(inarray=self.c2r_input_fft_array, outarray=self.c2r_output_fft_array, 
                                       direction='forward', flags=['measure']) 

        self._r2c_input_fft_array = np.zeros((self.data[0].shape[0]-1)*2, dtype=np.float64)
        self._r2c_output_fft_array = np.zeros(self.data[0].shape[0], dtype=np.complex128)
        self._r2c_fft_plan = fftw3.Plan(inarray=self.r2c_input_fft_array, outarray=self.r2c_output_fft_array, direction='forward', flags=['measure'])

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
    def msnr(self):
        """The SNR below which the prior goes to zero (or ``None`` for no threshold)."""
        return self._msnr

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

    @property 
    def c2r_input_fft_array(self):
        return self._c2r_input_fft_array

    @property
    def c2r_output_fft_array(self):
        return self._c2r_output_fft_array

    @property
    def c2r_fft_plan(self):
        return self._c2r_fft_plan

    @property
    def r2c_input_fft_array(self):
        return self._r2c_input_fft_array

    @property
    def r2c_output_fft_array(self):
        return self._r2c_output_fft_array

    @property
    def r2c_fft_plan(self):
        return self._r2c_fft_plan

    def generate_waveform(self, params):
        """Returns a frequency-domain strain suitable to subtract from the
        frequency-domain data (i.e. the samples line up in frequency
        space).
        """

        params = to_params(params)

        # Can only handle one parameter set at a time, so extract
        # first from array if more than one.
        if isinstance(params, np.ndarray) and params.ndim > 0:
            params = params[0]
        elif isinstance(params, np.ndarray):
            params = params[()]
        
        m1,m2 = u.mc_eta_to_masses(np.exp(params['log_mc']), params['eta'])
        d = 1e6*lal.LAL_PC_SI*np.exp(params['log_dist'])
        i = np.arccos(params['cos_iota'])

        dec = np.arcsin(params['sin_dec'])

        psi = params['psi']
        inc = np.arccos(params['cos_iota'])

        a1 = params['a1']
        tilt1 = np.arccos(params['cos_tilt1'])
        phi1 = params['phi1']
        a2 = params['a2']
        tilt2 = np.arccos(params['cos_tilt2'])
        phi2 = params['phi2']

        zhat = np.array([np.sin(inc), 0.0, np.cos(inc)])
        xhat = np.array([np.cos(inc), 0.0, -np.sin(inc)])
        yhat = np.array([0.0, 1.0, 0.0])

        s1 = a1 * (np.cos(phi1)*np.sin(tilt1)*xhat +\
                   np.sin(phi1)*np.sin(tilt1)*yhat +\
                   np.cos(tilt1)*zhat)
        s2 = a2 * (np.cos(phi2)*np.sin(tilt2)*xhat +\
                   np.sin(phi2)*np.sin(tilt2)*yhat +\
                   np.cos(tilt2)*zhat)

        print 'Params = ', params
        print 's1 = ', s1
        print 's2 = ', s2

        if ls.SimInspiralImplementedFDApproximants(self.approx) == 1:
            hplus,hcross = ls.SimInspiralChooseFDWaveform(params['phi'], 
                                                          self.fs[1]-self.fs[0],
                                                          m1*lal.LAL_MSUN_SI, m2*lal.LAL_MSUN_SI, 
                                                          s1[0], s1[1], s1[2],
                                                          s2[0], s2[1], s2[2],
                                                          self.fmin, 0.0,
                                                          d, i, 
                                                          0.0, 0.0,
                                                          None, None, 
                                                          self.amp_order, self.phase_order, 
                                                          self.approx)

            hpdata = hplus.data.data
            hcdata = hcross.data.data

            # If necessary, cut down to size
            if hpdata.shape[0] > self.fs.shape[0]:
                hpdata = hpdata[:self.fs.shape[0]]
                hcdata = hcdata[:self.fs.shape[0]]
        else:
            hplus,hcross = ls.SimInspiralChooseTDWaveform(params['phi'],
                                                          1.0/self.srate,
                                                          m1*lal.LAL_MSUN_SI, m2*lal.LAL_MSUN_SI, 
                                                          s1[0], s1[1], s1[2],
                                                          s2[0], s2[1], s2[2],
                                                          self.fmin, 0.0,
                                                          d, i, 
                                                          0.0, 0.0,
                                                          None, None, 
                                                          self.amp_order, self.phase_order, 
                                                          self.approx)
            
            Ntime = (self.data[0].shape[0]-1)*2
            
            # Cut down to length if necessary
            hpdata = hplus.data.data
            hcdata = hcross.data.data
            if hpdata.shape[0] > Ntime:
                hpdata = hpdata[-Ntime:]
                hcdata = hcdata[-Ntime:]

            # Now Fourier transiform; place the waveform's tC index
            # into the zero index of the FT array
            tC_index = int(round(-(hplus.epoch.gpsSeconds + 1e-9*hplus.epoch.gpsNanoSeconds)*self.srate))
            Nbegin = hpdata.shape[0] - tC_index

            self.r2c_input_fft_array[:] = 0.0
            self.r2c_input_fft_array[:Nbegin] = hpdata[tC_index:]
            self.r2c_input_fft_array[-tC_index:] = hpdata[:tC_index]
            self.r2c_fft_plan()
            hpdata = self.r2c_output_fft_array / self.srate # multiply by dt

            self.r2c_input_fft_array[:] = 0.0
            self.r2c_input_fft_array[:Nbegin] = hcdata[tC_index:]
            self.r2c_input_fft_array[-tC_index:] = hcdata[:tC_index]
            self.r2c_fft_plan()
            hcdata = self.r2c_output_fft_array / self.srate # multiply by dt

        N = hpdata.shape[0]

        hout=[]
        for d in self.detectors:
            tgps = lal.LIGOTimeGPS(0)
            
            sec = self.time_offset.sec + int(params['time'])
            ns = self.time_offset.ns + int(round(1e9*(params['time']-int(params['time']))))

            while ns > 1e9:
                sec += 1
                ns -= 1e9
                
            tgps.gpsSeconds = sec
            tgps.gpsNanoSeconds = ns            

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

            h = combine_and_timeshift(fplus, fcross, hpdata, hcdata, self.fs, timeshift)

            hout.append(h)

        return hout

    def malmquist_snr(self, params):
        """Returns the SNR that will be used in the Malmquist threshold in the
        likelihood function.

        The malmquist SNR is either:

        * The SNR in the second-loudest detector if there are two or
          more detectors.

        * The SNR if there is only one detector.

        The intention is to approximate a coincidence threshold from a
        pipeline.

        """

        hs = self.generate_waveform(params)
        df = self.fs[1] - self.fs[0]

        rhos = [np.sqrt(4.0*df*np.real(np.sum(np.conj(h)*h/self.psd))) for h in hs]
        
        if len(rhos) > 1:
            rhos.sort()
            return rhos[-2]
        else:
            return rhos[0]

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
            hh,dh = data_waveform_inner_product(istart, df, self.psd, h, d)

            hh_list.append(hh)

            logl += -0.5*(hh - 2.0*dh)

        # If malmquist priors, then cutoff when the SNR is too quiet.
        hh_list.sort()
        if self.msnr is not None:
            if len(hh_list) > 1 and hh_list[1] < self.msnr*self.msnr:
                return float('-inf')
            elif len(hh_list) == 1 and hh_list[0] < self.msnr*self.msnr:
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

        if params['phi'] > np.pi or params['phi'] < 0.0:
            return float('-inf')

        if params['psi'] > 2.0*np.pi or params['psi'] < 0.0:
            return float('-inf')

        if params['time'] < 0 or params['time'] > self.T:
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
        
        # Prior volumetric in distance:
        logp += 3.0*params['log_dist']

        return logp

    def draw_prior(self, shape=(1,)):
        params = np.zeros(shape, dtype=params_dtype)

        m1s = np.random.uniform(low=self.mmin, high=self.mmax, size=shape)
        m2s = np.random.uniform(low=self.mmin, high=self.mmax, size=shape)

        mc, eta = u.masses_to_mc_eta(m1s, m2s)

        params['log_mc'] = np.log(mc)
        params['eta'] = eta

        params['cos_iota'] = np.random.uniform(low=-1.0, high=1.0, size=shape)
        params['phi'] = np.random.uniform(low=0.0, high=np.pi, size=shape)
        params['psi'] = np.random.uniform(low=0.0, high=2.0*np.pi, size=shape)
        params['time'] = np.random.uniform(low=0.0, high=self.T, size=shape)
        params['ra'] = np.random.uniform(low=0.0, high=2.0*np.pi, size=shape)
        params['sin_dec'] = np.random.uniform(low=-1.0, high=1.0, size=shape)
        params['log_dist'] = np.log(self.dmax) + (1.0/3.0)*np.log(np.random.uniform(size=shape))

        return params

    def argmax_log_likelihood_tphid(self, params):
        params = to_params(params)

        df = self.fs[1] - self.fs[0]
        hs = self.generate_waveform(params)

        
        dh_dt_cos = 0.0
        dh_dt_sin = 0.0
        hh = 0.0
        for d, h in zip(self.data, hs):
            conj_d = np.conj(d)
            dh_real = 2.0*df*conj_d*np.real(h)/self.psd
            dh_imag = 2.0*df*conj_d*np.imag(h)/self.psd

            self.c2r_input_fft_array[:] = dh_real
            self.c2r_fft_plan()
            dh_dt_cos += self.c2r_output_fft_array

            self.c2r_input_fft_array[:] = dh_imag
            self.c2r_fft_plan()
            dh_dt_sin += self.c2r_output_fft_array

            hh += np.sum(4.0*df*np.abs(h)*np.abs(h)/self.psd)


        dh_dt = np.sqrt(dh_dt_cos*dh_dt_cos + dh_dt_sin*dh_dt_sin)
        idt = np.argmax(dh_dt)

        if idt == 0:
            a = np.abs(dh_dt[0])
            b = np.abs(dh_dt[1])
            c = np.abs(dh_dt[2])
            i0 = 0
        elif idt == len(dh_dt) - 1:
            a = np.abs(dh_dt[-3])
            b = np.abs(dh_dt[-2])
            c = np.abs(dh_dt[-1])
            i0 = len(dh_dt) - 3
        else:
            a = np.abs(dh_dt[idt-1])
            b = np.abs(dh_dt[idt])
            c = np.abs(dh_dt[idt+1])
            i0 = idt-1

        imax = i0 + 0.5 + (a-b)/(a+c-2.0*b)

        dt_max = imax/float(self.srate)
        dphi_max = -0.5*np.arctan2(dh_dt_sin[idt], dh_dt_cos[idt])

        dh_max = np.abs(dh_dt[idt])
        dfactor = hh / dh_max
        logd_max = params['log_dist'] + np.log(dfactor)

        max_params = params.copy()

        max_params['log_dist'] = logd_max

        max_params['phi'] = np.mod(params['phi'] + dphi_max, np.pi)
        if max_params['phi'] < 0:
            max_params['phi'] += np.pi

        max_params['time'] = np.mod(params['time'] + dt_max, self.T)

        return max_params

    def __call__(self, params):
        lp = self.log_prior(params)

        if lp == float('-inf'):
            return lp

        return lp + self.log_likelihood(params)

class TimeMarginalizedPosterior(Posterior):
    """Posterior that marginalizes out the time variable on each
    likelihood call."""

    def __init__(self, *args, **kwargs):
        """See :method:`Posterior.__init__`."""
        super(TimeMarginalizedPosterior, self).__init__(*args, **kwargs)

    def malmquist_snr(self, params):
        """See :method:`Posterior.malmquist_snr`."""
        p = time_marginalized_params_to_params(params, time=0)

        return super(TimeMarginalizedPosterior, self).malmquist_snr(p)

    def log_likelihood(self, params):
        """Returns the marginalized log-likelihood at the given params (which
        should have all parameters but time)."""
        
        params = to_time_marginalized_params(params)
        params_full = time_marginalized_params_to_params(params, time=0)

        hs = self.generate_waveform(params_full)

        ll = 0.0
        df = self.fs[1] - self.fs[0]
        dt = 1.0 / self.srate
        N_half = self.fs.shape[0]

        hh_list = []
        dh_timeshifts = 0.0
        for h, d in zip(hs, self.data):
            hh = hh_sum(df, self.psd, h)

            hh_list.append(hh)

            fill_fft_array(df, self.psd, d, h, self.c2r_input_fft_array)
            self.c2r_fft_plan()
            dh_timeshifts += self.c2r_output_fft_array
            
            ll += -0.5*hh

        dh = logaddexp_sum(dh_timeshifts)
        ll += dh

        # Normalization for time integral
        ll -= np.log(self.T)

        if self.msnr is not None:
            if len(hh_list) == 1:
                hh2nd = hh_list[0]
            else:
                hh_list.sort()
                hh2nd = hh_list[-2]

            if hh2nd < self.msnr*self.msnr:
                return float('-inf')

        return ll

    def log_prior(self, params):
        """Log prior; same as :method:`Posterior.log_prior`, but without
        `time` column.

        """
        
        params = to_time_marginalized_params(params)
        params_full = time_marginalized_params_to_params(params, time = 0.5*self.T)

        return super(TimeMarginalizedPosterior, self).log_prior(params_full)

    def draw_prior(self, shape=(1,)):
        pfull = super(TimeMarginalizedPosterior, self).draw_prior(shape=shape)
        return params_to_time_marginalized_params(pfull.reshape((-1,))).reshape(shape)

    def argmax_log_likelihood_phid(self, params):
        params_full = time_marginalized_params_to_params(params, time = 0.5*self.T)
        
        p = params_to_time_marginalized_params(super(TimeMarginalizedPosterior, self).argmax_log_likelihood_tphid(params_full))

        return p
