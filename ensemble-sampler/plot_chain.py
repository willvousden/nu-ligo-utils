#!/usr/bin/env python

from argparse import ArgumentParser
import matplotlib.pyplot as pp
import numpy as np
import plotutils.plotutils as pu
import posterior as pos
import utils as u

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--chain', metavar='FILE', required=True, help='input chain')
    parser.add_argument('--nwalkers', metavar='N', required=True, type=int, help='number of walkers')

    parser.add_argument('--inj-params', metavar='FILE', help='parameter file for injection')

    parser.add_argument('--noshow', default=False, action='store_true', help='do not show the plots after constructing them')

    parser.add_argument('--nothin', default=False, action='store_true', help='do not thin the chains first')

    args = parser.parse_args()

    chain = np.loadtxt(args.chain)
    chain = np.reshape(chain, (-1, args.nwalkers, chain.shape[-1]))
    
    if not args.nothin:
        chain = u.thin_chain(chain)

    chain = pos.to_params(chain.reshape((-1, chain.shape[-1]))).reshape((-1, args.nwalkers))

    if args.inj_params is not None:
        ip = pos.to_params(np.loadtxt(args.inj_params))
        if isinstance(ip, np.ndarray):
            ip = ip[0]
    else:
        ip = None
    
    for ln, (n, t) in zip(pos.params_latex, pos.params_dtype):
        pp.clf()
        pp.subplot(1,2,1)
        pu.plot_kde_posterior(chain[n].flatten())
        pp.xlabel('$' + ln + '$')
        pp.ylabel(r'$p\left(' + ln + r'\right)$')
        if ip is not None:
            pp.axvline(ip[n])

        pp.subplot(1,2,2)
        pp.plot(np.mean(chain[n], axis=0))
        pp.xlabel('Iteration')
        pp.ylabel('$' + ln + '$')
        if ip is not None:
            pp.axhline(ip[n])

        pp.savefig(n + '.pdf')

        if not args.noshow:
            pp.show()
