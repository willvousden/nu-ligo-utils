#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import emcee
import glob
import gzip
import lal
import lalsimulation as ls
import multiprocessing as multi
import numpy as np
import posterior as pos
import pylal.frutils as fu
import sys

class LogLikelihood(object):
    def __init__(self, lnpost):
        self.lnpost = lnpost

    def __call__(self, params):
        return self.lnpost.log_likelihood(params)

class LogPrior(object):
    def __init__(self, lnpost):
        self.lnpost = lnpost

    def __call__(self, params):
        return self.lnpost.log_prior(params)

class ArgmaxLogLikelihoodPhiD(object):
    def __init__(self, lnpost):
        self.lnpost = lnpost

    def __call__(self, params):
        return self.lnpost.argmax_log_likelihood_phid(params)

class MalmquistSNR(object):
    def __init__(self, lnpost):
        self.lnpost = lnpost

    def __call__(self, params):
        return self.lnpost.malmquist_snr(params)

# Workaround for stupid 2.6 that doesn't have gzip context manager
class GzipContextManager(object):
    def __init__(self, *args):
        self._args = args
        self._gzipfile = None

    def __enter__(self):
        self._gzipfile = gzip.open(*self._args)
        return self._gzipfile

    def __exit__(self, ex1, ex2, ex3):
        if self._gzipfile is not None:
            self._gzipfile.close()
            
        # Didn't handle any exceptions
        return False

def mywith_gzip_open(*args):
    return GzipContextManager(*args)

def reset_files(ntemps):
    for i in range(ntemps):
        with mywith_gzip_open('chain.{0:02d}.dat.gz'.format(i), 'r') as inp:
            header = inp.readline()
        with mywith_gzip_open('chain.{0:02d}.dat.gz'.format(i), 'w') as out:
            out.write(header)

        with mywith_gzip_open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'r') as inp:
            header = inp.readline()
        with mywith_gzip_open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'w') as out:
            out.write(header)

        with mywith_gzip_open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'r') as inp:
            header = inp.readline()
        with mywith_gzip_open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'w') as out:
            out.write(header)

def fix_malmquist(p0, lnposterior, rho_min, nthreads=1):
    """Returns a new set of parameters that satisfy the malmquist snr
    limit by choosing uniformly in allowed distances for any
    parameters that initially fail the malmquist test.

    :param p0: The initial parameter set ``(Ntemps, Nwalkers, Nparams)``

    :param lnposterior: The posterior object.

    :param rho_min: The Malmquist SNR limit.

    :param nthreads: The number of threads to use in computing the
      Malmquist SNR.

    """

    print 'Fixing up SNR\'s for Malmquist limits'
    print
    sys.stdout.flush()

    if args.nthreads > 1:
        pool = multi.Pool(args.nthreads)
        mm = pool.map
    else:
        mm = map

    msnr = MalmquistSNR(lnposterior)

    rhos = list(mm(msnr, p0.reshape((-1, nparams))))

    for rho, p in zip(rhos, p0.reshape((-1, nparams))):
        if rho < rho_min:
            p = lnposterior.to_params(p)
            d = np.exp(p['log_dist'])
            dmax = rho * d / args.malmquist_snr
            p['log_dist'] = np.log(dmax) + (1.0/3.0)*np.log(np.random.uniform())

    return p0
    

def recenter_best(chains, best, lnpost, malmquist_snr, shrinkfactor=10.0, nthreads=1):
    """Returns the given chains re-centered about the best point.

    :param chains: Shape ``(NTemps, NWalkers, NParams)``.

    :param best: The best point from the chain.

    :param lnpost: Log-posterior object (used to check prior bounds).

    :param shrinkfactor: The shrinkage factor in each dimension with
      respect to the spread of ``chain[0, :, :]``.

    """

    cov = np.cov(chains[0,:,:], rowvar = 0)
    cov /= shrinkfactor*shrinkfactor

    new_chains = np.random.multivariate_normal(best, cov, size=chains.shape[:-1])

    for i in range(new_chains.shape[0]):
        for j in range(new_chains.shape[1]):
            while lnpost.log_prior(new_chains[i,j,:]) == float('-inf'):
                new_chains[i,j,:] = np.random.multivariate_normal(best, cov)

    new_chains[0,0,:] = best

    if malmquist_snr is not None:
        new_chains = fix_malmquist(new_chains, lnpost, malmquist_snr, nthreads=nthreads)

    return new_chains

if __name__ == '__main__':
    parser = ArgumentParser(description='run an ensemble MCMC analysis of a GW event')

    parser.add_argument('--dataseed', metavar='N', type=int, help='seed for data generation')

    parser.add_argument('--ifo', metavar='IN', default=[], action='append', help='incorporate IFO in the analysis')
    parser.add_argument('--cache', metavar='CFILE', default=[], action='append', help='cache file associated with IFO')
    parser.add_argument('--channel', metavar='CHAN', default=[], action='append', help='channel assoicated with IFO strain in cache')

    parser.add_argument('--psd', metavar='PSD_FILE', help='file giving the PSDs in the analysis (one column per detector)')

    parser.add_argument('--data-start-sec', metavar='N', type=int, help='GPS integer seconds of data start')

    parser.add_argument('--seglen', metavar='DT', type=float, help='data segment length')

    parser.add_argument('--srate', metavar='R', default=16384.0, type=float, help='sample rate (in Hz)')

    parser.add_argument('--fmin', metavar='F', default=20.0, type=float, help='minimum frequency for integration/waveform')
    parser.add_argument('--fref', metavar='F', default=100.0, type=float, help='frequency at which time-dependent quantities are computed')

    parser.add_argument('--malmquist-snr', metavar='SNR', type=float, help='SNR threshold for Malmquist prior')

    parser.add_argument('--mmin', metavar='M', default=1.0, type=float, help='minimum component mass')
    parser.add_argument('--mmax', metavar='M', default=35.0, type=float, help='maximum component mass')
    parser.add_argument('--dmax', metavar='D', default=1000.0, type=float, help='maximim distance (Mpc)')

    parser.add_argument('--noise-only', action='store_true', help='run with only a noise model')

    parser.add_argument('--inj-xml', metavar='XML_FILE', help='LAL XML containing sim_inspiral table')
    parser.add_argument('--event', metavar='N', type=int, default=0, help='row index in XML table')

    parser.add_argument('--npsdfit', metavar='N', type=int, default=4, help='number of PSD fitting parameters')

    parser.add_argument('--start-position', metavar='FILE', help='file containing starting positions for chains')

    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of ensemble walkers')
    parser.add_argument('--nensembles', metavar='N', type=int, default=100, help='number of ensembles to accumulate')
    parser.add_argument('--nthin', metavar='N', type=int, default=10, help='number of setps to take between each saved ensemble state')

    parser.add_argument('--nthreads', metavar='N', type=int, default=1, help='number of concurrent threads to use')

    parser.add_argument('--Tmax', metavar='T', type=float, default=200.0, help='maximum temperature in the PT ladder')
    parser.add_argument('--ntemps', metavar='N', type=int, help='number of temperatures in PT ladder')

    parser.add_argument('--restart', default=False, action='store_true', help='continue a previously-existing run')

    parser.add_argument('--mpi', action='store_true', help='Run under MPI.')

    args=parser.parse_args()

    if len(args.ifo) == 0:
        args.ifo = ['H1', 'L1', 'V1']

    time_data = None
    if not (args.cache == []):
        time_data = []
        for cache, channel in zip(args.cache, args.channel):
            with open(cache, 'r') as inp:
                cache=fu.Cache.fromfile(inp)
                fcache=fu.FrameCache(cache)

                time_data.append(fcache.fetch(channel, args.data_start_sec, args.data_start_sec+args.seglen))

    if args.psd is not None:
        psd = list(np.transpose(np.loadtxt(args.psd)))
    else:
        psd = None

    # By default, start at GPS 0
    if args.data_start_sec is not None:
        gps_start = lal.LIGOTimeGPS(args.data_start_sec)
    else:
        gps_start = lal.LIGOTimeGPS(0)

    if args.noise_only:
        lnposterior = pos.NoiseOnlyPosterior(time_data=time_data,
                                             inj_xml=args.inj_xml,
                                             T=args.seglen,
                                             time_offset=gps_start,
                                             srate=args.srate,
                                             malmquist_snr=args.malmquist_snr,
                                             mmin=args.mmin,
                                             mmax=args.mmax,
                                             dmax=args.dmax,
                                             dataseed=args.dataseed,
                                             approx=ls.SpinTaylorT4,
                                             fmin=args.fmin,
                                             fref=args.fref,
                                             detectors=args.ifo,
                                             psd=psd,
                                             npsdfit=args.npsdfit)
        nparams = lnposterior.no_nparams
    else:
        lnposterior = pos.TimeMarginalizedPosterior(time_data=time_data,
                                                    inj_xml=args.inj_xml,
                                                    T=args.seglen,
                                                    time_offset=gps_start,
                                                    srate=args.srate,
                                                    malmquist_snr=args.malmquist_snr,
                                                    mmin=args.mmin,
                                                    mmax=args.mmax,
                                                    dmax=args.dmax,
                                                    dataseed=args.dataseed,
                                                    approx=ls.SpinTaylorT4,
                                                    fmin=args.fmin,
                                                    fref=args.fref,
                                                    detectors=args.ifo,
                                                    psd=psd,
                                                    npsdfit=args.npsdfit)
        nparams = lnposterior.tm_nparams


    pool = None
    if args.mpi:
        pool = MPIPool(loadbalance=True)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    sampler = emcee.PTSampler(None, args.nwalkers, nparams,
                              LogLikelihood(lnposterior),
                              LogPrior(lnposterior),
                              Tmax=args.Tmax, ntemps=args.ntemps,
                              threads=args.nthreads, pool=pool)

    Ts = 1.0/sampler.betas
    NTs = len(Ts)

    # Set up initial configuration
    p0 = np.zeros((NTs, args.nwalkers, nparams))
    means = []
    if args.restart:
        try:
            for i in range(NTs):
                p0[i, :, :] = np.loadtxt('chain.%02d.dat.gz'%i)[-args.nwalkers:,:]

            means = list(np.mean(np.loadtxt('chain.00.dat.gz').reshape((-1, args.nwalkers, nparams)), axis=1))
        except:
            for i in range(NTs):
                p0[i,:,:] = lnposterior.draw_prior(shape=(args.nwalkers,)).view(float).reshape((args.nwalkers, nparams))

            if args.malmquist_snr is not None:
                fix_malmquist(p0, lnposterior, args.malmquist_snr, nthreads=args.nthreads)
            means = []
            
            # Since we failed to load the old chain, don't restart
            args.restart = False
    elif args.start_position is not None:
        p0 = np.loadtxt(args.start_position).reshape((NTs, args.nwalkers, nparams))
        
        if args.malmquist_snr is not None:
            fix_malmquist(p0, lnposterior, args.malmquist_snr, nthreads=args.nthreads)
    else:
        for i in range(NTs):
            p0[i,:,:] = lnposterior.draw_prior(shape=(args.nwalkers,)).view(float).reshape((args.nwalkers, nparams))

        if args.malmquist_snr is not None:
            fix_malmquist(p0, lnposterior, args.malmquist_snr, nthreads=args.nthreads)

    np.savetxt('temperatures.dat', Ts.reshape((1,-1)))
    with open('sampler-params.dat', 'w') as out:
        out.write('# NTemps NWalkers Nthin\n')
        out.write('{0:d} {1:d} {2:d}\n'.format(NTs, args.nwalkers, args.nthin))

    freq_data_columns = (lnposterior.fs,)
    for d in lnposterior.data:
        freq_data_columns = freq_data_columns + (np.real(d), np.imag(d))
    np.savetxt('freq-data.dat.gz', np.column_stack(freq_data_columns))

    with open('command-line.txt', 'w') as out:
        out.write(' '.join(sys.argv) + '\n')

    # Set up headers:
    if not args.restart:
        for i in range(NTs):
            with mywith_gzip_open('chain.%02d.dat.gz'%i, 'w') as out:
                header = lnposterior.header
                out.write('# ' + header + '\n')

            with mywith_gzip_open('chain.%02d.lnlike.dat.gz'%i, 'w') as out:
                out.write('# lnlike0 lnlike1 ...\n')

            with mywith_gzip_open('chain.%02d.lnpost.dat.gz'%i, 'w') as out:
                out.write('# lnpost0 lnpost1 ...\n')

    print 'Beginning ensemble evolution.'
    print
    sys.stdout.flush()

    lnpost = None
    lnlike = None
    old_best_lnlike = None
    reset = False
    while True:
        for p0, lnpost, lnlike in sampler.sample(p0, lnprob0=lnpost, lnlike0=lnlike, iterations=args.nthin, storechain=False):
            pass

        print 'afrac = ', ' '.join(map(lambda x: '{0:6.3f}'.format(x), np.mean(sampler.acceptance_fraction, axis=1)))
        print 'tfrac = ', ' '.join(map(lambda x: '{0:6.3f}'.format(x), sampler.tswap_acceptance_fraction))
        sys.stdout.flush()
        
        if reset:
            reset_files(NTs)
            reset = False
        for i in range(NTs):
            with mywith_gzip_open('chain.{0:02d}.dat.gz'.format(i), 'a') as out:
                np.savetxt(out, p0[i,:,:])
            with mywith_gzip_open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'a') as out:
                np.savetxt(out, lnlike[i,:].reshape((1,-1)))
            with mywith_gzip_open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'a') as out:
                np.savetxt(out, lnpost[i,:].reshape((1,-1)))

        maxlnlike = np.max(lnlike)
            
        if old_best_lnlike is None:
            old_best_lnlike = maxlnlike
            
            if not args.restart:
                # If on first iteration, then start centered around
                # the best point so far
                imax = np.argmax(lnlike)
                best = p0.reshape((-1, p0.shape[-1]))[imax,:]
                p0 = recenter_best(p0, best, lnposterior, args.malmquist_snr, shrinkfactor=10.0, nthreads=args.nthreads)

                lnpost = None
                lnlike = None
                sampler.reset()

        if maxlnlike > old_best_lnlike + p0.shape[-1]/2.0:
            old_best_lnlike = maxlnlike
            reset = True
            means = []

            imax = np.argmax(lnlike)
            best = p0.reshape((-1, p0.shape[-1]))[imax,:]
            p0 = recenter_best(p0, best, lnposterior, args.malmquist_snr, shrinkfactor=10.0, nthreads=args.nthreads)
            
            # And reset the log(L) values
            lnpost = None
            lnlike = None
            sampler.reset()

            print 'Found new best likelihood of {0:.1f}.'.format(old_best_lnlike)
            print 'Resetting around parameters '
            best_params = lnposterior.to_params(best).squeeze()
            for n in best_params.dtype.names:
                print n + ':', best_params[n]
            print 
            sys.stdout.flush()

        means.append(np.mean(p0[0, :, :], axis=0))

        ameans = np.array(means)
        ameans = ameans[int(round(0.2*ameans.shape[0])):, :]
        taumax = float('-inf')
        for j in range(ameans.shape[1]):
            try:
                tau = acor.acor(ameans[:,j])[0]
            except:
                tau = float('inf')

            taumax = max(tau, taumax)

        ndone = int(round(ameans.shape[0]/taumax))

        print 'Computed {0:d} effective ensembles (max correlation length is {1:g})'.format(ndone, taumax)
        print
        sys.stdout.flush()

        if ndone > args.nensembles:
            break
