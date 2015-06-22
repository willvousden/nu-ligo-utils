#!/usr/bin/env python

from argparse import ArgumentParser
import glob
import matplotlib.pyplot as plt
import numpy as np

def ti_evidence(logls, betas, plotout):
    r"""Returns the thermodynamic integration :math:`\ln(Z)` from the given
    log-likelihoods and inverse temperatures.

    :param logls: The log-likelihoods.  Should be of shape
      ``(ntemperatures, nsamples, nwalkers)``.

    :param betas: The array of temperatures.

    :param plotout: If not ``None``, save a plot of the contribution
      to the evidence integral versus ``T``.

    :return: ``(ln_Z, delta_ln_Z)`` a measure of both the evidence and
      the uncertainty in the integration of the evidence.

    """

    betas_zero = np.array(list(betas) + [0.0])
    betas2_zero = np.array(list(betas[::2]) + [0.0])
    dbetas = -np.diff(betas_zero)
    dbetas2 = -np.diff(betas2_zero)

    mean_logl = np.mean(logls.reshape((logls.shape[0], -1)), axis=1)

    integrand = mean_logl*dbetas

    if plotout is not None:
        plt.clf()
        plt.plot(1.0/betas, integrand)
        plt.xscale('log')
        plt.xlabel(r'$T$')
        plt.ylabel(r'$\left\langle \log \mathcal{L} \right\rangle_\beta d\beta$')
        plt.savefig(plotout)

    ln_Z = np.sum(integrand)
    dln_Z = np.abs(ln_Z - np.sum(mean_logl[::2]*dbetas2))

    return ln_Z, dln_Z

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input', metavar='STEM', default='chain', help='stem of input files')
    parser.add_argument('--temps', metavar='FILE', default='temperatures.dat', help='temperature file')
    parser.add_argument('--fburnin', metavar='F', default=0.2, type=float, help='fraction of samples to discard as burnin')
    parser.add_argument('--output', metavar='FILE', help='output file')

    parser.add_argument('--plotout', metavar='FILE', help='plot the integral contributions versus T')

    args = parser.parse_args()

    files = glob.glob(args.input + '.[0-9][0-9].lnlike.dat.gz')
    files.sort()
    logls = []

    for file in files:
        logls.append(np.loadtxt(file))
    min_len = reduce(min, map(lambda ll: ll.shape[0], logls))
    logls = [ll[:min_len,:] for ll in logls]

    logls = np.array(logls)

    iburnin = int(round(args.fburnin*logls.shape[1]))
    logls = logls[:, iburnin:, :]

    betas = 1.0/np.loadtxt(args.temps)

    lnZ, dlnZ = ti_evidence(logls, betas, args.plotout)

    if args.output is not None:
        with open(args.output, 'w') as out:
            out.write('# ln(Z) d(ln(Z))\n')
            np.savetxt(out, np.array([lnZ, dlnZ]).reshape((1,-1)))

    print('Evidence is {0:.1f} +/- {1:.1f}'.format(lnZ, dlnZ))
    
