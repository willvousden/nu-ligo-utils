#!/usr/bin/env python

from argparse import ArgumentParser
import glob
import numpy as np

def tail(f, n):
    """Returns the last ``n`` lines of the file ``f``."""

    i = 0
    lines = []
    for l in f:
        if i < n:
            lines.append(l)
            i = i+1
        else:
            lines[i%n] = l
            i = i+1

    if i < n:
        raise ValueError('tail: file must have at least {0:d} lines'.format(n))

    return lines[i%n:] + lines[:i%n]

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--nwalkers', metavar='N', default=100, type=int, help='number of walkers')

    args = parser.parse_args()

    for f in glob.glob('chain.[0-9][0-9].dat'):
        with open(f, 'r') as inp:
            header = inp.readline()
            lines = tail(inp, args.nwalkers)

        with open(f, 'w') as out:
            out.write(header)
            for l in lines:
                out.write(l) # Comes with '\n'

    for f in glob.glob('chain.[0-9][0-9].lnlike.dat') + glob.glob('chain.[0-9][0-9].lnpost.dat'):
        with open(f, 'r') as inp:
            header = inp.readline()
            lines = tail(inp, 1)

        with open(f, 'w') as out:
            out.write(header + '\n')
            out.write(lines[0])
