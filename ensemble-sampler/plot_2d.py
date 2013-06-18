#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pp
import os.path as op
import params 
import plotutils.plotutils as pu

def plot_2d(chain, inj, nx, ny, lnx, lny, outdir, show):
    pp.clf()

    pu.plot_histogram_posterior_2d(chain[[nx,ny]].view(float).reshape((-1,2)))

    if inj is not None:
        pp.plot(inj[nx], inj[ny], '*r')

    pp.xlabel('$' + lnx + '$')
    pp.ylabel('$' + lny + '$')

    pp.savefig(op.join(outdir, '{0:s}-{1:s}.pdf'.format(nx,ny)))
    
    if show:
        pp.show()

def plot_1d(chain, inj, name, lname, outdir, show):
    pp.clf()

    pu.plot_histogram_posterior(chain[name].view(float).flatten(), normed=True, histtype='step')
    
    if inj is not None:
        pp.axvline(inj[name])

    pp.xlabel('$' + lname + '$')
    pp.ylabel(r'$p\left( ' + lname + r' \right)$')

    pp.savefig(op.join(outdir, '{0:s}.pdf'.format(name)))

    if show:
        pp.show()

def plot_chain(chain, inj, name, lname, outdir, show):
    pp.clf()

    pp.plot(np.mean(chain[name], axis=1))
    
    if inj is not None:
        pp.axhline(inj[name])

    pp.xlabel('iteration')
    pp.ylabel(r'$\left \langle ' + lname + r' \right\rangle$')

    pp.savefig(op.join(outdir, '{0:s}-chain.pdf'.format(name)))

    if show:
        pp.show()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--chain', metavar='FILE', help='chain file', required=True)
    parser.add_argument('--nwalkers', metavar='N', type=int, help='number of walkers', required=True)

    parser.add_argument('--inj', metavar='FILE', help='injection file')

    parser.add_argument('--outdir', metavar='DIR', default='.', help='output directory')

    parser.add_argument('--fburnin', metavar='F', type=float, help='fraction to discard as burnin')

    parser.add_argument('--noshow', action='store_false', help='do not show plots')
    
    args = parser.parse_args()

    chain = pu.load_header_data(args.chain, header_commented=True).reshape((-1, args.nwalkers))
    if args.fburnin is not None:
        ibin = int(round(args.fburnin*chain.shape[0]))
        chain = chain[ibin:,:]
    
    if args.inj:
        inj = pu.load_header_data(args.inj, header_commented=True)
    else:
        inj = None

    names = [n for (n,t) in params.params_time_marginalized_dtype]
    lnames = params.params_time_marginalized_latex
    N = len(names)

    for i in range(N):
        plot_1d(chain, inj, names[i], lnames[i], args.outdir, args.noshow)
        plot_chain(chain, inj, names[i], lnames[i], args.outdir, args.noshow)
        for j in range(i+1,N):
            plot_2d(chain, inj, names[i], names[j], lnames[i], lnames[j], args.outdir, args.noshow)
