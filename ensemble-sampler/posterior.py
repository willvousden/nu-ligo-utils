import numpy as np
import fftw3
import lal
import lalsimulation as ls
from posterior_utils import *
from pylal import SimInspiralUtils
import scipy.interpolate as si
import scipy.stats as st
import utils as u

class Posterior(object):
    """Callable object representing the posterior."""

    def __init__(self, time_data=None, freq_data=None,
                 inj_params=None, inj_xml=None, event=0, srate=16384,
                 T=None, time_offset=lal.LIGOTimeGPS(0),
                 approx=ls.TaylorF2, amp_order=-1, phase_order=-1,
                 fmin=20.0, fref=100.0, malmquist_snr=None, mmin=1.0,
                 mmax=35.0, dmax=1000.0, dataseed=None,
                 data_psdparams=None, detectors=['H1', 'L1', 'V1'],
                 psd=None, npsdfit=4):
        r"""Set up the posterior.  Currently only does PE on H1 with iIGOIGO
        analytic noise spectrum.

        :param time_data: A list of float arrays giving the
          time-domain data in each detector on which the analysis is
          to operate.  If ``None``, then data are generated from
          Gaussian noise.  The time-domain data will be windowed with
          the default Tukey window from :func:`u.tukey_window` before
          being Fourier-transformed.

        :param freq_data: A list of complex arrays giving the
          frequency-domain data in each detector on which the analysis
          is to operate.  If both ``time_data`` and ``freq_data`` are
          ``None`` then data are generated from Gaussian noise.

        :param inj_params: Parameters for a waveform to be injected.

        :param inj_xml: XML filename describing a waveform to be
          injected.

        :param event: The event number (starting with zero) of the
          injection from the XML.

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

        :param fref: The reference frequency where freq-dependent
          waveform quantities are computed.

        :param malmquist_snr: If not ``None``, gives the SNR threshold
          in the second-loudest detector (or only detector) below
          which the prior probability is zero.

        :param mmin: Minimum component mass threshold.

        :param mmax: Maximum component mass threshold.
        
        :param dmax: Maximum distance.

        :param dataseed: If not ``None``, will be used as a RNG seed
          for generating any synthetic data.

        :param data_psdparams: If not ``None``, the PSD fitting
          parameters to be used to modify the PSD when producing
          synthetic data.  This argument only makes sense when both
          ``time_data`` and ``freq_data`` are ``None``.

        :param detectors: The detectors on which the analysis runs.

        :param psd: A list of PSDs to use instead of the synthetic
          AdLIGO PSD from LALSimultion.  There should be one PSD per
          detector.

        :param npsdfit: The number of PSD fitting parameters to use.

        """

        self._srate = srate
        self._time_offset = u.GPSTime(time_offset.gpsSeconds, time_offset.gpsNanoSeconds)
        self._approx = approx
        self._amp_order = amp_order
        self._phase_order = phase_order
        self._fmin = fmin
        self._fref = fref
        self._msnr = malmquist_snr
        self._mmin = mmin
        self._mmax = mmax
        self._dmax = dmax
        self._detectors = detectors

        if T is None:
            self._T = time_data[0].shape[0]/srate
        else:
            self._T = T

        data_length = int(round(self.T*self.srate/2+1))

        self._fs = np.linspace(0, srate/2.0, self.T*self.srate/2+1)

        self._npsdfit = npsdfit
        self._psdfitfs = np.exp(np.linspace(np.log(self.fmin), np.log(self.fs[-1]), self.npsdfit))

        if psd is not None:
            # Cut the PSD down to length if it's too long
            self._psd = [p[:self.fs.shape[0]] for p in psd]
        else:
            self._psd = [np.zeros(self.fs.shape[0]) for d in detectors]
            for d, psd in zip(detectors, self.psd):
                if d[0] == 'H' or d[0] == 'L':
                    for i in range(self.fs.shape[0]):
                        psd[i] = ls.SimNoisePSDaLIGOZeroDetHighPower(self.fs[i])
                elif d[0] == 'V':
                    for i in range(self.fs.shape[0]):
                        psd[i] = ls.SimNoisePSDAdvVirgo(self.fs[i])

        # Zero out PSD below fmin
        for p in self.psd:
            p[self.fs < fmin] = float('inf')

        if time_data is None and freq_data is None:
            self._data = [np.zeros(data_length, dtype=np.complex) for d in detectors]

            # Maybe set seed?
            if dataseed is not None:
                old_state = np.random.get_state()
                np.random.seed(dataseed)
            
            if data_psdparams is not None:
                params = np.zeros(1, dtype=self.dtype)
                params['psdfit'] = data_psdparams
                psd = self.adjusted_psd(params)
            else:
                psd = self.psd

            for j in range(len(detectors)):
                # 0.5 = 2 * 1/sqrt(2).  One sqrt(2) from
                # one-sided-->two-sided, and the other from <|z|> =
                # sqrt(2) if x,y ~ N(0,1).
                self.data[j] = 0.5*np.sqrt(psd[j]/(self.fs[1]-self.fs[0]))*(np.random.normal(size=data_length) +
                                                                            np.random.normal(size=data_length)*1j)
                self.data[j][psd[j]==float('inf')] = 0.0

            # Reset random state
            if dataseed is not None:
                np.random.set_state(old_state)
        elif time_data is None:
            self._data = freq_data
        else:
            self._data = []
            for i in range(len(detectors)):
                N = time_data[i].shape[0]

                this_srate = float(N)/self.T
                dt = 1.0/this_srate

                window = u.tukey_window(N)

                fdata = np.fft.rfft(time_data[i]*window)*dt

                # Now cut down to the actual sample rate
                self._data.append(fdata[:data_length])

        self._c2r_input_fft_array = np.zeros(self.data[0].shape[0], dtype=np.complex128)
        self._c2r_output_fft_array = np.zeros((self.data[0].shape[0]-1)*2, dtype=np.float64)
        self._c2r_fft_plan = fftw3.Plan(inarray=self.c2r_input_fft_array, outarray=self.c2r_output_fft_array, 
                                       direction='forward', flags=['measure']) 

        self._r2c_input_fft_array = np.zeros((self.data[0].shape[0]-1)*2, dtype=np.float64)
        self._r2c_output_fft_array = np.zeros(self.data[0].shape[0], dtype=np.complex128)
        self._r2c_fft_plan = fftw3.Plan(inarray=self.r2c_input_fft_array, outarray=self.r2c_output_fft_array, direction='forward', flags=['measure'])

        if inj_xml is not None:
            params = self.inj_xml_to_params(inj_xml)
            hs = self.generate_waveform(params)
            for i, h in enumerate(hs):
                self.data[i] += h
        elif inj_params is not None:
            hs = self.generate_waveform(inj_params)
            for i,h in enumerate(hs):
                self.data[i] += h
        
    # Handle unpickling the internal state
    def __setstate__(self, state):
        for k,v in state.items():
            self.__dict__[k] = v

        # Just the FFTW3 Plans are screwed up:
        self._r2c_fft_plan = fftw3.Plan(inarray=self.r2c_input_fft_array, outarray=self.r2c_output_fft_array, direction='forward', flags=['measure'])
        self._c2r_fft_plan = fftw3.Plan(inarray=self.c2r_input_fft_array, outarray=self.c2r_output_fft_array, direction='forward', flags=['measure']) 

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
    def df(self):
        """The spacing in frequency space.

        """
        return self.fs[1]-self.fs[0]

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
    def fref(self):
        """The reference frequency at which freq-dependent waveform quantities
        are defined."""

        return self._fref

    @property
    def psd(self):
        """The array of (one-sided) noise PSDs used in the analysis (one per
        detector)."""
        return self._psd

    @property
    def npsdfit(self):
        """The number of PSD fitting parameters to use.

        """

        return self._npsdfit

    @property
    def psdfitfs(self):
        """The frequencies at which the PSD fit spline knots live.

        """

        return self._psdfitfs

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
    def ndetectors(self):
        return len(self.detectors)

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

    @property
    def nparams(self):
        """The dimensionality of the parameter space."""
        return 15 + self.ndetectors*self.npsdfit

    @property
    def header(self):
        """A useful header describing the parameters for this posterior in text files.

        """

        header = ['log_mc', 'eta', 'cos_iota', 'phi', 'psi', 'time', 'ra',
                  'sin_dec', 'log_dist', 'a1', 'cos_tilt1', 'phi1', 'a2', 'cos_tilt2', 
                  'phi2']

        for d in self.detectors:
            for i in range(self.npsdfit):
                header.append('{0:s}PSD{1:02d}'.format(d,i))

        return ' '.join(header)

    def to_params(self, p):
        return p.view([('log_mc', np.float),
                       ('eta', np.float),
                       ('cos_iota', np.float),
                       ('phi', np.float),
                       ('psi', np.float),
                       ('time', np.float),
                       ('ra', np.float),
                       ('sin_dec', np.float),
                       ('log_dist', np.float),                    
                       ('a1', np.float),
                       ('cos_tilt1', np.float),
                       ('phi1', np.float),
                       ('a2', np.float),
                       ('cos_tilt2', np.float),
                       ('phi2', np.float)] + [('psdfit', np.float, (self.ndetectors, self.npsdfit))])

    def adjusted_psd(self, params):
        """Returns a PSD for each detector, adjusted by the PSD parameters for
        that detector.

        """

        if self.npsdfit == 0:
            return self.psd

        params = self.to_params(params)
        sel = self.fs >= self.fmin

        fs = self.fs[sel]

        psds = []

        for raw_psd, psdp in zip(self.psd, params['psdfit'].squeeze()):
            log_factors = si.InterpolatedUnivariateSpline(np.log(self.psdfitfs), psdp)(np.log(fs))

            psd = raw_psd.copy()
            psd[sel] *= np.exp(log_factors)

            psds.append(psd)

        return psds

    def inj_xml_to_params(self, inj_xml, event=0, psdfit=None):
        """Returns the parameters that correspond to the given XML file,
        optionally with the given PSD fitting parameters.

        :param inj_xml: Filename of the injection XML.

        :param event: The event number to use from the XML.

        :param psdfit: The PSD fitting parameters to add to the
          returned parameters.

        """

        p = self.to_params(np.zeros(self.nparams))

        table = SimInspiralUtils.ReadSimInspiralFromFiles([inj_xml])[event]

        p['log_mc'] = np.log(table.mchirp)
        p['eta'] = table.eta
        p['log_dist'] = np.log(table.distance)
        p['ra'] = table.longitude
        p['sin_dec'] = np.sin(table.latitude)
        p['cos_iota'] = np.cos(table.inclination)
        p['phi'] = table.coa_phase
        p['psi'] = table.polarization
    
        time_offset = self.time_offset.LIGOTimeGPS
        p['time'] = table.geocent_end_time - time_offset.gpsSeconds + 1e-9*(table.geocent_end_time_ns - time_offset.gpsNanoSeconds)

        s1 = np.array([table.spin1x, table.spin1y, table.spin1z])
        s2 = np.array([table.spin2x, table.spin2y, table.spin2z])

        Lhat = np.array([np.sin(table.inclination), 0.0, np.cos(table.inclination)])
        xhat = np.array([np.cos(table.inclination), 0.0, -np.sin(table.inclination)])
        yhat = np.array([0.0,1.0,0.0])

        if np.linalg.norm(s1) == 0.0:
            p['a1'] = 0.0
            p['cos_tilt1'] = 1.0
            p['phi1'] = 0.0
        else:
            a1 = np.linalg.norm(s1)
            p['a1'] = a1
            p['cos_tilt1'] = np.dot(s1, Lhat)/a1
            p['phi1'] = np.arctan2(np.dot(s1, yhat), np.dot(s1, xhat))
            if p['phi1'] < 0.0:
                p['phi1'] += 2.0*np.pi

        if np.linalg.norm(s2) == 0.0:
            p['a2'] = 0.0
            p['cos_tilt2'] = 1.0
            p['phi2'] = 0.0
        else:
            a2 = np.linalg.norm(s2)
            p['a2'] = a2
            p['cos_tilt2'] = np.dot(s2, Lhat)/a2
            p['phi2'] = np.arctan2(np.dot(s2, yhat), np.dot(s2, xhat))
            if p['phi2'] < 0.0:
                p['phi2'] += 2.0*np.pi

        if psdfit is not None:
            p['psdfit'] = psdfit

        return p


    def generate_waveform(self, params):
        """Returns a frequency-domain strain suitable to subtract from the
        frequency-domain data (i.e. the samples line up in frequency
        space).
        """

        params = self.to_params(params).squeeze()

        # Can only handle one parameter set at a time, so extract
        # first from array if more than one.
        if isinstance(params, np.ndarray) and params.ndim > 0:
            params = params[0]
        elif isinstance(params, np.ndarray):
            params = params[()]
        
        m1,m2 = u.mc_eta_to_masses(np.exp(params['log_mc']), params['eta'])
        d = 1e6*lal.PC_SI*np.exp(params['log_dist'])
        i = np.arccos(params['cos_iota'])

        dec = np.arcsin(params['sin_dec'])

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

        if ls.SimInspiralImplementedFDApproximants(self.approx) == 1:
            hplus,hcross = ls.SimInspiralChooseFDWaveform(params['phi'], 
                                                          self.fs[1]-self.fs[0],
                                                          m1*lal.MSUN_SI, m2*lal.MSUN_SI, 
                                                          s1[0], s1[1], s1[2],
                                                          s2[0], s2[1], s2[2],
                                                          self.fmin, self.fs[-1], self.fref,
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
                                                          m1*lal.MSUN_SI, m2*lal.MSUN_SI, 
                                                          s1[0], s1[1], s1[2],
                                                          s2[0], s2[1], s2[2],
                                                          self.fmin, self.fref,
                                                          d, i, 
                                                          0.0, 0.0,
                                                          None, None, 
                                                          self.amp_order, self.phase_order, 
                                                          self.approx)
            
            Ntime = (self.data[0].shape[0]-1)*2
            
            # Cut down to length if necessary
            hpdata = hplus.data.data
            hcdata = hcross.data.data
            tC_index = int(round(-(hplus.epoch.gpsSeconds + 1e-9*hplus.epoch.gpsNanoSeconds)*self.srate))
            if hpdata.shape[0] > Ntime:
                tC_index -= hpdata.shape[0] - Ntime
                hpdata = hpdata[-Ntime:]
                hcdata = hcdata[-Ntime:]

            # Now Fourier transiform; place the waveform's tC index
            # into the zero index of the FT array
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

        hout=[]
        for d in self.detectors:
            sec = self.time_offset.sec + int(params['time'])
            ns = self.time_offset.ns + int(round(1e9*(params['time']-int(params['time']))))

            while ns > 1e9:
                sec += 1
                ns -= 1e9
                
            tgps = lal.LIGOTimeGPS(sec, nanoseconds=ns)

            gmst = lal.GreenwichMeanSiderealTime(tgps)

            if d == 'H1':
                diff = lal.LALDetectorIndexLHODIFF
            elif d == 'L1':
                diff = lal.LALDetectorIndexLLODIFF
            elif d == 'V1':
                diff = lal.LALDetectorIndexVIRGODIFF
            else:
                raise ValueError('detector not recognized: ' + d)
            
            location = lal.CachedDetectors[diff].location

            timedelay = lal.TimeDelayFromEarthCenter(location, params['ra'], dec, tgps)

            timeshift = params['time'] + timedelay
                
            fplus, fcross = lal.ComputeDetAMResponse(lal.CachedDetectors[diff].response,
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

        adj_psd = self.adjusted_psd(params)
        rhos = [np.sqrt(4.0*df*np.real(np.sum(np.conj(h)*h/psd))) for h, psd in zip(hs, adj_psd)]
        
        if len(rhos) > 1:
            rhos.sort()
            return rhos[-2]
        else:
            return rhos[0]

    def log_likelihood(self, params):
        r"""Returns the log likelihood of the given parameters.  The
log-likelihood is

        .. math::
          
          \log \mathcal{L} = -\frac{1}{2} \left( \left\langle d | d \right\rangle -2 \Re \left\langle d | h \right\rangle + \left\langle h | h \right\rangle \right) - \frac{1}{2} \sum \log S(f)

        where 

        .. math::

          \left\langle a | b \right\rangle = 4 \int df \, \frac{a^*(f) b(f)}{S(f)},

        where :math:`S(f)` is the one-sided noise power spectral density.

        This corresponds to the ususal log-likelihood in Gaussian
        noise, accounting for the fact that parameters can cause the
        PSD to vary.

        If the :attr:`Posterior.malmquest_snr` is not ``None``, then
        the likelihood will be returned as ``float('-inf')`` when
        :math:`\left\langle h | h \right\rangle^{1/2}` is smaller than
        :attr:`Posterior.malmquest_snr`"""

        hs = self.generate_waveform(params)
        df = self.fs[1] - self.fs[0]

        istart = np.nonzero(self.fs >= self.fmin)[0][0]

        hh_list=[]
        logl = 0.0
        adj_psd = self.adjusted_psd(params)
        for h, d, psd in zip(hs, self.data, adj_psd):
            hh,dh,dd = data_waveform_inner_product(istart, df, psd, h, d)

            hh_list.append(hh)

            logl += -0.5*(hh - 2.0*dh + dd)
            logl -= np.sum(np.log(2.0*np.pi*psd[istart:]/(4.0*(self.fs[1]-self.fs[0]))))

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
        params = self.to_params(params)

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

        if params['time'] < 0 or params['time'] > self.T:
            return float('-inf')

        if params['ra'] < 0.0 or params['ra'] > 2.0*np.pi:
            return float('-inf')

        if params['sin_dec'] < -1.0 or params['sin_dec'] > 1.0:
            return float('-inf')

        if d > self.dmax:
            return float('-inf')

        if params['a1'] <= 0.0 or params['a1'] >= 1.0:
            return float('-inf')

        if params['a2'] <= 0.0 or params['a2'] >= 1.0:
            return float('-inf')

        if params['cos_tilt1'] < -1.0 or params['cos_tilt1'] > 1.0:
            return float('-inf')

        if params['cos_tilt2'] < -1.0 or params['cos_tilt2'] > 1.0:
            return float('-inf')

        if params['phi1'] < 0.0 or params['phi1'] >= 2.0*np.pi:
            return float('-inf')

        if params['phi2'] < 0.0 or params['phi2'] >= 2.0*np.pi:
            return float('-inf')

        logp = 0.0

        # A flat prior in mass space gives the following in log(mc)-eta space:
        logp -= np.log(m1-m2) - 3.0*np.log(mtot)
        
        # Prior volumetric in distance:
        logp += 3.0*params['log_dist']

        # N(0,1) prior on PSD parameters (which are log(factor) at
        # each frequency).
        logp += np.sum(u.norm_logpdf(params['psdfit']))

        if isinstance(logp, np.ndarray):
            if logp.ndim > 0:
                logp = logp[0]
            else:
                logp = logp[()]

        return logp

    def draw_prior(self, shape=(1,)):
        params = self.to_params(np.zeros(shape+(self.nparams,))).squeeze()

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
        params['a1'] = np.random.uniform(low=0.0, high=1.0, size=shape)
        params['a2'] = np.random.uniform(low=0.0, high=1.0, size=shape)
        params['cos_tilt1'] = np.random.uniform(low=-1.0, high=1.0, size=shape)
        params['cos_tilt2'] = np.random.uniform(low=-1.0, high=1.0, size=shape)
        params['phi1'] = np.random.uniform(low=0.0, high=2.0*np.pi, size=shape)
        params['phi2'] = np.random.uniform(low=0.0, high=2.0*np.pi, size=shape)

        params['psdfit'] = np.random.normal(size=shape + (len(self.detectors), self.npsdfit))

        return params

    def argmax_log_likelihood_tphid(self, params):
        params = self.to_params(params)

        df = self.fs[1] - self.fs[0]
        hs = self.generate_waveform(params)

        
        dh_dt_cos = 0.0
        dh_dt_sin = 0.0
        hh = 0.0
        adj_psd = self.adjusted_psd(params)

        for d, h, psd in zip(self.data, hs, adj_psd):
            conj_d = np.conj(d)
            dh_real = 2.0*df*conj_d*np.real(h)/psd
            dh_imag = 2.0*df*conj_d*np.imag(h)/psd

            self.c2r_input_fft_array[:] = dh_real
            self.c2r_fft_plan()
            dh_dt_cos += self.c2r_output_fft_array

            self.c2r_input_fft_array[:] = dh_imag
            self.c2r_fft_plan()
            dh_dt_sin += self.c2r_output_fft_array

            hh += np.sum(4.0*df*np.abs(h)*np.abs(h)/psd)


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

    def to_params(self, params):
        try:
            return params.view([('log_mc', np.float),
                                ('eta', np.float),
                                ('cos_iota', np.float),
                                ('phi', np.float),
                                ('psi', np.float),
                                ('ra', np.float),
                                ('sin_dec', np.float),
                                ('log_dist', np.float),                    
                                ('a1', np.float),
                                ('cos_tilt1', np.float),
                                ('phi1', np.float),
                                ('a2', np.float),
                                ('cos_tilt2', np.float),
                                ('phi2', np.float)] + \
                               [('psdfit', np.float, (self.ndetectors, self.npsdfit))])
        except:
            return super(TimeMarginalizedPosterior, self).to_params(params)

    @property
    def header(self):
        """A useful header describing the parameters for this posterior in text files.

        """

        header = ['log_mc', 'eta', 'cos_iota', 'phi', 'psi', 'ra', 'sin_dec', 
                  'log_dist', 'a1', 'cos_tilt1', 'phi1', 'a2', 'cos_tilt2', 
                  'phi2']

        for d in self.detectors:
            for i in range(self.npsdfit):
                header.append('{0:s}PSD{1:02d}'.format(d,i))

        return ' '.join(header)

    @property
    def tm_nparams(self):
        return 14 + self.ndetectors*self.npsdfit

    def to_super_params(self, params, time=0):
        params = self.to_params(params)
        sps = super(TimeMarginalizedPosterior, self).to_params(np.zeros(params.shape + (self.nparams,)))

        for name in params.dtype.names:
            sps[name] = params[name]

        sps['time'] = time

        return sps

    def from_super_params(self, params):
        params = self.to_params(params)
        sps = self.to_params(np.zeros(params.shape+(self.tm_nparams,))).squeeze()

        for name in sps.dtype.names:
            sps[name] = params[name]

        return sps

    def adjusted_psd(self, params):
        pfull = self.to_super_params(params)
        return super(TimeMarginalizedPosterior, self).adjusted_psd(pfull)

    def malmquist_snr(self, params):
        """See :method:`Posterior.malmquist_snr`."""
        p = self.to_super_params(params, time=0)

        return super(TimeMarginalizedPosterior, self).malmquist_snr(p)

    def time_integrate(self, log_ls):
        """Returns the log of the integral of the given log(L) values as a
        function of time, using an analytic, quadratic interpolation
        of the log(L) values.

        """
        
        full_log_ls = np.zeros(log_ls.shape[0]+1)
        full_log_ls[:-1] = log_ls
        full_log_ls[-1] = log_ls[0]

        dt = 1.0/self.srate

        # dt*sum(log_ls) = dt*(1/2*fll[0] + fll[1] + ... + fll[N-1] + 1/2*fll[N])
        # This is the trapezoid rule for the integral.
        log_best_integral = logaddexp_sum(log_ls) + np.log(dt)

        return log_best_integral

    def log_likelihood(self, params):
        """Returns the marginalized log-likelihood at the given params (which
        should have all parameters but time)."""
        
        params = self.to_params(params)
        params_full = self.to_super_params(params, time=0)

        hs = self.generate_waveform(params_full)

        ll = 0.0
        df = self.fs[1] - self.fs[0]

        hh_list = []
        dh_timeshifts = 0.0
        adj_psd = self.adjusted_psd(params)
        for h, d, psd in zip(hs, self.data, adj_psd):
            hh,dd = hh_dd_sum(df, psd, h, d)

            hh_list.append(hh)

            fill_fft_array(df, psd, d, h, self.c2r_input_fft_array)
            self.c2r_fft_plan()
            dh_timeshifts += self.c2r_output_fft_array
            
            ll += -0.5*(hh + dd)
            ll -= np.sum(np.log(2.0*np.pi*psd[psd != float('inf')]/(4.0*(self.fs[1]-self.fs[0]))))

        dh = self.time_integrate(dh_timeshifts)
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
        
        params = self.to_params(params)
        params_full = self.to_super_params(params, time = 0.5*self.T)

        return super(TimeMarginalizedPosterior, self).log_prior(params_full)

    def draw_prior(self, shape=(1,)):
        pfull = super(TimeMarginalizedPosterior, self).draw_prior(shape=shape)
        return self.from_super_params(pfull)

    def argmax_log_likelihood_phid(self, params):
        params_full = self.to_super_params(params, time = 0.5*self.T)
        
        p = self.from_super_params(super(TimeMarginalizedPosterior, self).argmax_log_likelihood_tphid(params_full))

        return p

class NoiseOnlyPosterior(Posterior):
    """Represents the posterior for a noise-only model.

    """

    def __init__(self, *args, **kwargs):
        super(NoiseOnlyPosterior, self).__init__(*args, **kwargs)

    @property
    def header(self):
        header = []
        for d in self.detectors:
            for i in range(self.npsdfit):
                header.append('{0:s}PSD{1:02d}'.format(d,i))

        return ' '.join(header)


    @property
    def no_nparams(self):
        return self.ndetectors*self.npsdfit

    def to_params(self, params):
        try:
            return params.view([('psdfit', np.float, (self.ndetectors, self.npsdfit))])
        except:
            return super(NoiseOnlyPosterior, self).to_params(params)

    def generate_waveform(self, params):
        if params.view(float).shape[0] == self.no_nparams:
            hs = []
            for d in self.data:
                hs.append(0.0*d)
            return hs
        else:
            return super(NoiseOnlyPosterior, self).generate_waveform(params)

    def log_prior(self, params):
        return np.sum(u.norm_logpdf(params))

    def draw_prior(self, shape=(1,)):
        return self.to_params(np.random.normal(size=shape+(self.ndetectors*self.npsdfit,)))

        
