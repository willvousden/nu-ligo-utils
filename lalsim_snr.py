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

fhigh_fudgefactor = 1.1
max_srate = 4096.

def nextPow2(length):
    """
    Find next power of 2 <= length
    """
    return int(2**np.ceil(np.log2(length)))

def unwind_phase(phase,thresh=5.):
    """
    Unwind an array of values of a periodic variable so that it does not jump
    discontinuously when it hits the periodic boundary, but changes smoothly
    outside the periodic range.

    Note: 'thresh', which determines if a discontinuous jump occurs, should be
    somewhat less than the periodic interval. Empirically, 5 is usually a safe
    value of thresh for a variable with period 2 pi.
    """
    cnt = 0 # count number of times phase wraps around branch cut
    length = len(phase)
    unwound = np.zeros(length)
    unwound[0] = phase[0]
    for i in range(1,length):
        if phase[i-1] - phase[i] > thresh: # phase wrapped forward
            cnt += 1
        elif phase[i] - phase[i-1] > thresh: # phase wrapped backward
            cnt -= 1
        unwound[i] = phase[i] + cnt * 2. * np.pi
    return unwound

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
  
    nyquist = 2*(f_isco*fhigh_fudgefactor*(1.+temp_amp_order))

    seglen_fudgefactor = 1.1
    chirptime =  seglen_fudgefactor * lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(f_low, mass1, mass2, chi, phase_order)
  
    seglen = nextPow2(chirptime)
  
    if calcSNR:
        segStart = event.geocent_end_time-seglen+2
        deltaF = 1./seglen

        # First generate the worst case with a high sampling rate to determine a sampling rate
        srate = max_srate
        deltaT = 1./srate
      
        hp,hc = lalsim.SimInspiralChooseTDWaveform(
            event.coa_phase,
            1.0/srate,
            mass1, mass2,
            0., 0., 1.,
            0., 0., 1.,
            f_low_inj, 0,
            event.distance * 1.0e6 * lal.LAL_PC_SI,
            0.,
            0, 0, None, None,
            0, phase_order,
            approx)

        hc = hp.data.data + 1j * hc.data.data
        ph = unwind_phase(np.angle(hc))
        f_stop = abs(ph[-2] - ph[-1]) / (deltaT * 2 * np.pi)
        nyquist = 2 * f_stop * fhigh_fudgefactor * (1 + temp_amp_order)

        srate = nextPow2(nyquist)

        if srate > max_srate:
            print "WARNING: Sampling rate is not sufficient for the highest frequency contributions of the waveform.  Falling back to 4096 Hz since the noise is probably too high to measure this anyways."
            srate = max_srate
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

      
        networkSNR = 0.0
        for ifo in ifos:
            h = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, event.longitude, event.latitude, event.polarization, lalsim.DetectorPrefixToLALDetector(ifo))

            td_len = h.data.length
            fd_len = td_len // 2 + 1

            h_tilde = lal.CreateCOMPLEX16FrequencySeries("h_tilde", h.epoch, h.f0, 1./h.deltaT/td_len, lal.lalHertzUnit, fd_len)

            plan = lal.CreateForwardREAL8FFTPlan(td_len, 0)
            lal.REAL8TimeFreqFFT(h_tilde, h, plan)
      
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
    nyquist = 2*(f_isco*fhigh_fudgefactor*(1+amp_order))
  
    srate = nextPow2(nyquist)
  
    mass1 *= lal.LAL_MSUN_SI
    mass2 *= lal.LAL_MSUN_SI
    seglen_fudgefactor = 1.1
    chirptime =  seglen_fudgefactor * lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(f_low, mass1, mass2, 0., phase_order)
  
    seglen = nextPow2(chirptime)
  
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
