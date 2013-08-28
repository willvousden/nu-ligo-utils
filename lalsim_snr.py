#!/usr/bin/env python

import numpy as np

import lal
import lalsimulation as lalsim
from glue.ligolw import lsctables
from glue.ligolw import ligolw
from scipy import interpolate as interp
from pylal import SimInspiralUtils

# Hardcoded lalsim max pN orders
max_nonprecessing_amp_pn_order = 5
max_precessing_amp_pn_order = 3
max_phase_order = 7

class _vectorize_swig_psd_func(object):
    """Create a vectorized Numpy function from a SWIG-wrapped PSD function.
    SWIG does not provide enough information for Numpy to determine the number
    of input arguments, so we can't just use np.vectorize.  Stolen from Bayestar"""

    def __init__(self, func):
        self._npyfunc = np.frompyfunc(func, 1, 1)

    def __call__(self, f):
        ret = self._npyfunc(f)
        if not np.isscalar(ret):
            ret = ret.astype(float)
        return ret

def get_inj_info(temp_amp_order, inj, event=0, ifos=['H1','L1','V1'], era='advanced', f_low=30., calcSNR=True, psd_files=None):
    noise_psd_funcs = {}
    if psd_files is not None:
        for ifo in ifos:
            psd_data = np.loadtxt(psd_files[ifo])
            psd_data[:,1] = psd_data[:,1]*psd_data[:,1]
            psd = interp.interp1d(psd_data[:,0], psd_data[:,1])
            noise_psd_funcs[ifo] = psd

    else:
        if era == 'initial':
            for _ifos, _func in (
              (("H1", "H2", "L1", "I1"), lalsim.SimNoisePSDiLIGOModel),
              (("V1",), lalsim.SimNoisePSDVirgo)):
                _func = _vectorize_swig_psd_func(_func)
                for _ifo in _ifos:
                    noise_psd_funcs[_ifo] = _func
      
        elif era == 'advanced':
            for _ifos, _func in (
              (("H1", "H2", "L1", "I1"), lalsim.SimNoisePSDaLIGOZeroDetHighPower),
              (("V1",), lalsim.SimNoisePSDAdvVirgo),
              (("K1",), lalsim.SimNoisePSDKAGRA)):
                _func = _vectorize_swig_psd_func(_func)
                for _ifo in _ifos:
                      noise_psd_funcs[_ifo] = _func

    # Determine SNR of injection if given
    event = SimInspiralUtils.ReadSimInspiralFromFiles([inj])[event]
    phase_order = lalsim.GetOrderFromString(str(event.waveform))

    approx = lalsim.GetApproximantFromString(str(event.waveform))
    if approx is lalsim.TaylorF2:
        approx=lalsim.TaylorT4
        print "Frequency domain injections haven't been implemented in this script yet.  Calculating SNR with TaylorT4 (should be good enough)"
  
    # Nyquist for highest harmonic
    mass1 = event.mass1 * lal.LAL_MSUN_SI
    mass2 = event.mass2 * lal.LAL_MSUN_SI
    chi = lalsim.SimIMRPhenomBComputeChi(mass1, mass2, event.spin1z, event.spin2z)
  
    if temp_amp_order is None or temp_amp_order == -1:
        temp_amp_order = max_precessing_amp_pn_order if approx == lalsim.SpinTaylorT4 else max_nonprecessing_amp_pn_order
  
    f_low_inj = event.f_lower

    # f_low is the lower frequency to be used for integration, thus the starting frequency of the 2,2 mode is
    #  lower for waveforms with higher harmonics
    f_low_restricted = f_low * 2 / float(temp_amp_order+2)
    f_isco = 1.0 / (6.0 * np.sqrt(6.0) * np.pi * (event.mass1+event.mass2) * lal.LAL_MTSUN_SI)
  
    fhigh_fudgefactor = 1.1
    nyquist = 2*(f_isco*fhigh_fudgefactor*(1.+temp_amp_order))
    srate = 1.0
    while srate < nyquist: srate *= 2.
  
    seglen_fudgefactor = 1.1
    chirptime =  seglen_fudgefactor * lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(f_low, mass1, mass2, chi, phase_order)
  
    seglen = 1.0
    while seglen < chirptime: seglen *= 2.
  
    if calcSNR:
        segStart = event.geocent_end_time-seglen+2
        deltaF = 1./seglen
        deltaT = 1./srate
      
        hp,hc = lalsim.SimInspiralChooseTDWaveform(
            event.coa_phase,
            1.0/srate,
            mass1, mass2,
            event.spin1x, event.spin1y, event.spin1z,
            event.spin2x, event.spin2y, event.spin2z,
            f_low_inj, 0,
            event.distance * 1.0e6 * lal.LAL_PC_SI,
            event.inclination,
            0, 0, None, None,
            event.amp_order, phase_order,
            approx)
        lenF = hp.data.length // 2 + 1
      
        networkSNR = 0.0
        for ifo in ifos:
            h = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, event.longitude, event.latitude, event.polarization, lalsim.DetectorPrefixToLALDetector(ifo))
            h_tilde = lal.CreateCOMPLEX16FrequencySeries("h_tilde", lal.LIGOTimeGPS(0), 0, deltaF, lal.lalDimensionlessUnit, lenF)
            plan = lal.CreateForwardREAL8FFTPlan(len(hp.data.data), 0)
            lal.REAL8TimeFreqFFT(h_tilde, hp, plan)
      
            psd = noise_psd_funcs[ifo]
            freqs = np.arange(h_tilde.f0, h_tilde.data.length*h_tilde.deltaF, h_tilde.deltaF)
            freq_sel = np.where((freqs > f_low_inj) & (freqs < nyquist))
            np.seterr(divide='ignore')
      
            low = int(f_low_inj/h_tilde.deltaF)
            high = h_tilde.data.length
            SNR = np.sum(np.power(h_tilde.data.data.real[freq_sel],2)/psd(freqs[freq_sel]))
            SNR += np.sum(np.power(h_tilde.data.data.imag[freq_sel],2)/psd(freqs[freq_sel]))
            SNR *= 4.*h_tilde.deltaF
            networkSNR+=SNR

        networkSNR = np.sqrt(networkSNR)
        networkSNR /= 2.  # Why!?
    else:
        networkSNR=None
    return networkSNR, srate, seglen, f_low_restricted


def get_bns_info(f_low=30):
    # Nyquist for highest harmonic
    mass1 = mass2 = 1.4
    phase_order = max_phase_order
    amp_order = max_nonprecessing_amp_pn_order
  
    f_low_restricted = f_low * 2 / float(amp_order+2)
    f_isco = 1.0 / (6.0 * np.sqrt(6.0) * np.pi * (mass1+mass2) * lal.LAL_MTSUN_SI)
    fhigh_fudgefactor = 1.1
    nyquist = 2*(f_isco*fhigh_fudgefactor*(1+amp_order))
  
    srate = 1.0
    while srate < nyquist: srate *= 2.
  
    mass1 *= lal.LAL_MSUN_SI
    mass2 *= lal.LAL_MSUN_SI
    seglen_fudgefactor = 1.1
    chirptime =  seglen_fudgefactor * lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(f_low, mass1, mass2, 0., phase_order)
  
    seglen = 1.0
    while seglen < chirptime: seglen *= 2.
  
    return srate, seglen, f_low_restricted 


if __name__ == '__main__':
    import argparse
  
    parser = argparse.ArgumentParser(description='Calculate SNR of event in XML for a given analytic PSD')
  
    parser.add_argument('--inj', help='Injection XML file.')
    parser.add_argument('--event', default=0, type=int, help='Event number in XML to inject.')
    parser.add_argument('--era', default='initial',  help='Detector era for PSDs')
    parser.add_argument('--ifo', default=['H1','L1','V1'], action='append', help='IFOs for the analysis.')
    parser.add_argument('--flow', default=30.0, type=float, help='starting frequency')
  
    args = parser.parse_args()
  
    SNR, srate, seglen, f_low_restricted = get_inj_info(-1, args.inj, args.event, args.ifo, args.era, args.flow)
    print "Injection SNR: {:.2f}".format(SNR)
    print "Sampling Rate: {}".format(srate)
    print "Segment Length: {}".format(seglen)
    print "Required flow: {}".format(f_low_restricted)
