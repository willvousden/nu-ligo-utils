#!/usr/bin/env python

from argparse import ArgumentParser
import glob
import numpy as np

def ti_evidence(logls, betas):
    r"""Returns the thermodynamic integration :math:`\ln(Z)` from the given
    log-likelihoods and inverse temperatures.

    :param logls: The log-likelihoods.  Should be of shape
      ``(ntemperatures, nsamples, nwalkers)``.

    :param betas: The array of temperatures.

    :return: ``(ln_Z, delta_ln_Z)`` a measure of both the evidence and
      the uncertainty in the integration of the evidence.

    """

    betas_zero = np.array(list(betas) + [0.0])
    betas2_zero = np.array(list(betas[::2]) + [0.0])
    dbetas = -np.diff(betas_zero)
    dbetas2 = -np.diff(betas2_zero)

    mean_logl = np.mean(logls.reshape((logls.shape[0], -1)), axis=1)

    ln_Z = np.sum(mean_logl*dbetas)
    dln_Z = np.abs(ln_Z - np.sum(mean_logl[::2]*dbetas2))

    return ln_Z, dln_Z

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input', metavar='STEM', default='chain', help='stem of input files')
    parser.add_argument('--temps', metavar='FILE', default='temperatures.dat', help='temperature file')
    parser.add_argument('--fburnin', metavar='F', default=0.2, type=float, help='fraction of samples to discard as burnin')

    args = parser.parse_args()

    files = glob.glob(args.input + '.[0-9][0-9].lnlike.dat')
    files.sort()
    logls = []

    for file in files:
        logls.append(np.loadtxt(file))
    logls = np.array(logls)

    iburnin = int(round(args.fburnin*logls.shape[1]))
    logls = logls[:, iburnin:, :]

    betas = 1.0/np.loadtxt(args.temps)

    print ti_evidence(logls, betas)
    
