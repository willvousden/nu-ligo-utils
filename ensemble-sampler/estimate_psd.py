#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import pylal.frutils as fu
import scipy.signal as ss
import utils as u

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--cache', metavar='FILE', help='cache file', required=True)
    parser.add_argument('--channel', metavar='CHAN', help='channel name', required=True)
    parser.add_argument('--start', metavar='T', type=int, help='start GPS time', required=True)
    parser.add_argument('--end', metavar='T', type=int, help='end GPS time', required=True)
    parser.add_argument('--seglen', metavar='T', type=int, help='length of individual segment', required=True)
    parser.add_argument('--output', metavar='FILE', help='output file', required=True)

    args = parser.parse_args()

    with open(args.cache, 'r') as inp:
        cache = fu.Cache.fromfile(inp)
        fcache = fu.FrameCache(cache)

        data = fcache.fetch(args.channel, args.start, args.end)

    srate = 1.0/data.metadata.dt
    window = u.tukey_window(args.seglen*srate)
    psd = ss.welch(data, fs=1.0/data.metadata.dt, window=window)

    with open(args.output, 'w') as out:
        out.write('# f PSD(f)\n')
        np.savetxt(out, np.column_stack(psd))
    
