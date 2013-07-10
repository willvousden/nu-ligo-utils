#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import os.path
import params

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--chain', metavar='FILE', required=True, help='chain file')
    parser.add_argument('--output', metavar='FILE', default='posterior_samples.dat', help='output file')
    parser.add_argument('--nwalkers', metavar='N', type=int, help='number of walkers')
    parser.add_argument('--fburnin', metavar='F', type=float, help='fraction of samples to discard as burnin')
    parser.add_argument('--thin', metavar='N', type=int, help='thinning step for output samples')

    args = parser.parse_args()

    chain_base, ext = os.path.splitext(args.chain)
    logl_file = chain_base + '.lnlike' + ext
    logp_file = chain_base + '.lnpost' + ext

    with open(args.chain, 'r') as inp:
        chain_header = inp.readline().split()[1:]
        chain = np.loadtxt(inp)

    chain_header = [label.replace('_', '') for label in chain_header]

    # Now drop the PSD parameters
    chain_header = chain_header[:-1]
    chain = chain[:, :len(chain_header)]

    logls = np.loadtxt(logl_file).reshape((-1,1))
    logps = np.loadtxt(logp_file).reshape((-1,1))

    if args.fburnin is not None:
        assert args.nwalkers is not None, 'need to specify number of walkers for burnin'
        chain = chain.reshape((-1, args.nwalkers, chain.shape[1]))
        logls = logls.reshape((-1, args.nwalkers))
        logps = logps.reshape((-1, args.nwalkers))

        nchain = chain.shape[0]
        istart = int(round(nchain*args.fburnin))

        chain = chain[istart:, ...]
        logls = logls[istart:,:]
        logps = logps[istart:,:]

        chain = chain.reshape((-1, chain.shape[2]))
        logls = logls.reshape((-1, 1))
        logps = logps.reshape((-1, 1))
    
    if args.thin is not None:
        chain = chain.reshape((-1, args.nwalkers, chain.shape[1]))
        logls = logls.reshape((-1, args.nwalkers))
        logps = logps.reshape((-1, args.nwalkers))

        chain = chain[::args.thin, ...]
        logls = logls[::args.thin, :]
        logps = logps[::args.thin, :]

        chain = chain.reshape((-1, chain.shape[2]))
        logls = logls.reshape((-1, 1))
        logps = logps.reshape((-1, 1))

    min_n = min(chain.shape[0], logls.shape[0], logps.shape[0])
    chain = chain[:min_n,:]
    logls = logls[:min_n]
    logps = logps[:min_n]

    with open(args.output, 'w') as out:
        out.write(' '.join(chain_header + ['logl', 'logpost']) + '\n')
        np.savetxt(out, np.column_stack((chain, logls, logps)))
        

