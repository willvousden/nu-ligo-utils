#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import gzip
import numpy as np
import os.path

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--chain', metavar='FILE', required=True, help='chain file')
    parser.add_argument('--output', metavar='FILE', default='posterior_samples.dat', help='output file')
    parser.add_argument('--nwalkers', metavar='N', type=int, help='number of walkers')
    parser.add_argument('--fburnin', metavar='F', type=float, help='fraction of samples to discard as burnin')
    parser.add_argument('--thin', metavar='N', type=int, help='thinning step for output samples')
    parser.add_argument('--decorrelated', action='store_true', help='use only decorrelated samples')

    args = parser.parse_args()

    chain_base, gz_ext = os.path.splitext(args.chain)
    if gz_ext == '.gz':
        chain_base, ext = os.path.splitext(chain_base)
        logl_file = chain_base + '.lnlike' + ext + gz_ext
        logp_file = chain_base + '.lnpost' + ext + gz_ext
    else:
        logl_file = chain_base + '.lnlike' + gz_ext
        logp_file = chain_base + '.lnpost' + gz_ext

    if gz_ext == '.gz':
        o = gzip.open
    else:
        o = open
    with o(args.chain, 'r') as inp:
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

    if args.decorrelated:
        chain = chain.reshape((-1, args.nwalkers, chain.shape[1]))
        logls = logls.reshape((-1, args.nwalkers))
        logps = logps.reshape((-1, args.nwalkers))

        maxtau = float('-inf')
        
        for k in range(chain.shape[2]):
            maxtau = max(maxtau, acor.acor(np.mean(chain[:,:,k], axis=1))[0])
        maxtau = int(round(maxtau))

        chain = chain[::maxtau, ...]
        logls = logls[::maxtau, :]
        logps = logps[::maxtau, :]

    min_n = min(chain.shape[0], logls.shape[0], logps.shape[0])
    chain = chain[:min_n,:]
    logls = logls[:min_n]
    logps = logps[:min_n]

    out_base, ext = os.path.splitext(args.output)
    if ext == '.gz':
        o = gzip.open
    else:
        o = open
    with o(args.output, 'w') as out:
        out.write(' '.join(chain_header + ['logl', 'logpost']) + '\n')
        np.savetxt(out, np.column_stack((chain, logls, logps)))
        

