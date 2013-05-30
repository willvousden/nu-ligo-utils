#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import os.path
import params

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--chain', metavar='FILE', required=True, help='chain file')
    parser.add_argument('--output', metavar='FILE', default='posterior_samples.dat', help='output file')

    args = parser.parse_args()

    chain_base, ext = os.path.splitext(args.chain)
    logl_file = chain_base + '.lnlike' + ext
    logp_file = chain_base + '.lnpost' + ext

    with open(args.chain, 'r') as inp:
        chain_header = inp.readline().split()[1:]
        chain = np.loadtxt(inp)

    chain_header = [label.replace('_', '') for label in chain_header]

    logls = np.loadtxt(logl_file).reshape((-1,1))
    logps = np.loadtxt(logp_file).reshape((-1,1))

    with open(args.output, 'w') as out:
        out.write(' '.join(chain_header + ['logl', 'logpost']) + '\n')
        np.savetxt(out, np.column_stack((chain, logls, logps)))
        

