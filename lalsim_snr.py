#!/usr/bin/env python

import numpy as np

import lal
import lalsimulation as lalsim
from glue.ligolw import lsctables
from glue.ligolw import ligolw
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

def get_inj_info(temp_amp_order, inj, event=0, ifos=['H1','L1','V1'], era='advanced', intended_flow=30):
    noise_psd_funcs = {}
    if era == 'initial':
        for _ifos, _func in (
          (("H1", "H2", "L1", "I1"), lalsim.SimNoisePSDiLIGOModel),
          (("V1",), lalsim.SimNoisePSDVirgo)):
            _func = _vectorize_swig_psd_func(_func)
            for _ifo in _ifos:
                noise_psd_funcs[_ifo] = _func
  
    elif era is 'advanced':
        for _ifos, _func in (
          (("H1", "H2", "L1", "I1"), lalsim.SimNoisePSDaLIGOZeroDetHighPower),
          (("V1",), lalsim.SimNoisePSDAdvVirgo),
          (("K1",), lalsim.SimNoisePSDKAGRA)):
            _func = _vectorize_swig_psd_func(_func)
            for _ifo in _ifos:
                  noise_psd_funcs[_ifo] = _func
  
    # Determine SNR of injection if given
    event = SimInspiralUtils.ReadSimInspiralFromFiles([inj])[event]
    approx = lalsim.GetApproximantFromString(str(event.waveform))
    phase_order = lalsim.GetOrderFromString(str(event.waveform))
  
    # Nyquist for highest harmonic
    mass1 = event.mass1 * lalsim.lal.LAL_MSUN_SI
    mass2 = event.mass2 * lalsim.lal.LAL_MSUN_SI
    chi = lalsim.SimIMRPhenomBComputeChi(mass1, mass2, event.spin1z, event.spin2z)
  
    if temp_amp_order is None or temp_amp_order == -1:
        temp_amp_order = max_precessing_amp_pn_order if approx == lalsim.SpinTaylorT4 else max_nonprecessing_amp_pn_order
  
    flow = event.f_lower
    flow_hm = intended_flow / (1.+temp_amp_order)
    f_isco = 1.0 / (6.0 * np.sqrt(6.0) * np.pi * (event.mass1+event.mass2) * lalsim.lal.LAL_MTSUN_SI)
  
    fhigh_fudgefactor = 1.1
    nyquist = 2*(f_isco*fhigh_fudgefactor*(1.+temp_amp_order))
    srate = 1.0
    while srate < nyquist: srate *= 2.
  
    seglen_fudgefactor = 1.1
    chirptime =  seglen_fudgefactor * lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(intended_flow, mass1, mass2, chi, phase_order)
  
    seglen = 1.0
    while seglen < chirptime: seglen *= 2.
  
    segStart = event.geocent_end_time-seglen+2
    deltaF = 1./seglen
    deltaT = 1./srate
  
    hp,hc = lalsim.SimInspiralChooseTDWaveform(
        event.coa_phase,
        1.0/srate,
        mass1, mass2,
        event.spin1x, event.spin1y, event.spin1z,
        event.spin2x, event.spin2y, event.spin2z,
        flow, 0,
        event.distance * 1.0e6 * lalsim.lal.LAL_PC_SI,
        event.inclination,
        0, 0, None, None,
        event.amp_order, phase_order,
        approx)
    lenF = hp.data.length // 2 + 1
  
    networkSNR = 0.0
    for ifo in ifos:
        h = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, event.longitude, event.latitude, event.polarization, lalsim.DetectorPrefixToLALDetector(ifo))
        h_tilde = lalsim.lal.CreateCOMPLEX16FrequencySeries("h_tilde", lalsim.lal.LIGOTimeGPS(0), 0, deltaF, lalsim.lal.lalDimensionlessUnit, lenF)
        plan = lalsim.lal.CreateForwardREAL8FFTPlan(len(hp.data.data), 0)
        lalsim.lal.REAL8TimeFreqFFT(h_tilde, hp, plan)
  
        psd = noise_psd_funcs[ifo]
        freqs = np.arange(h_tilde.f0, h_tilde.data.length*h_tilde.deltaF, h_tilde.deltaF)
        np.seterr(divide='ignore')
  
        SNR = np.sum([np.power(h_tilde.data.data.real[j],2)/psd(freqs[j]) for j in xrange(int(flow/h_tilde.deltaF), h_tilde.data.length)])
        SNR += np.sum([np.power(h_tilde.data.data.imag[j],2)/psd(freqs[j]) for j in xrange(int(flow/h_tilde.deltaF), h_tilde.data.length)])
        SNR *= 4.*h_tilde.deltaF
        networkSNR+=SNR

    networkSNR = np.sqrt(networkSNR)
    networkSNR /= 2.  # Why!?
    return networkSNR, srate, seglen, flow_hm


def get_bns_info(intended_flow=30):
    # Nyquist for highest harmonic
    mass1 = mass2 = 1.4
    phase_order = max_phase_order
    amp_order = max_nonprecessing_amp_pn_order
  
    flow = intended_flow
    flow_hm = intended_flow / (1+amp_order)
    f_isco = 1.0 / (6.0 * np.sqrt(6.0) * np.pi * (mass1+mass2) * lalsim.lal.LAL_MTSUN_SI)
    fhigh_fudgefactor = 1.1
    nyquist = 2*(f_isco*fhigh_fudgefactor*(1+amp_order))
  
    srate = 1.0
    while srate < nyquist: srate *= 2.
  
    mass1 *= lalsim.lal.LAL_MSUN_SI
    mass2 *= lalsim.lal.LAL_MSUN_SI
    seglen_fudgefactor = 1.1
    chirptime =  seglen_fudgefactor * lalsim.SimInspiralTaylorF2ReducedSpinChirpTime(flow, mass1, mass2, 0., phase_order)
  
    seglen = 1.0
    while seglen < chirptime: seglen *= 2.
  
    return srate, seglen, flow_hm


if __name__ == '__main__':
    import argparse
  
    parser = argparse.ArgumentParser(description='Calculate SNR of event in XML for a given analytic PSD')
  
    parser.add_argument('--inj', help='Injection XML file.')
    parser.add_argument('--event', default=0, type=int, help='Event number in XML to inject.')
    parser.add_argument('--era', default='initial',  help='Detector era for PSDs')
    parser.add_argument('--ifo', default=['H1','L1','V1'], action='append', help='IFOs for the analysis.')
  
    args = parser.parse_args()
  
    SNR, srate, seglen = get_inj_info(args.inj, args.event, args.ifo, args.era)
    print "Injection SNR: {:.2f}".format(SNR)
    print "Sampling Rate: {}".format(srate)
    print "Segment Length: {}".format(seglen)
    print "Required flow: {}".format(flow_hm)
