#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import emcee
import lal
import lalsimulation as ls
import multiprocessing as multi
import numpy as np
import posterior as pos
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

def reset_files(ntemps):
    for i in range(ntemps):
        with open('chain.{0:02d}.dat'.format(i), 'r') as inp:
            header = inp.readline()
        with open('chain.{0:02d}.dat'.format(i), 'w') as out:
            out.write(header)

        with open('chain.{0:02d}.lnlike.dat'.format(i), 'r') as inp:
            header = inp.readline()
        with open('chain.{0:02d}.lnlike.dat'.format(i), 'w') as out:
            out.write(header)

        with open('chain.{0:02d}.lnpost.dat'.format(i), 'r') as inp:
            header = inp.readline()
        with open('chain.{0:02d}.lnpost.dat'.format(i), 'w') as out:
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

    # Make sure to clear the pool, so we don't have inactive
    # processes hanging around.
    mm = None
    pool = None

    for rho, p in zip(rhos, p0.reshape((-1, nparams))):
        if rho < rho_min:
            p = params.to_time_marginalized_params(p)
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

    parser.add_argument('--data', metavar='FILE', help='file containing time-domain strain data for analysis')
    parser.add_argument('--data-start-sec', metavar='N', type=int, help='GPS integer seconds of data start')
    parser.add_argument('--data-start-ns', metavar='N', type=int, help='GPS nano-seconds of data start')

    parser.add_argument('--seglen', metavar='DT', type=float, help='data segment length')

    parser.add_argument('--srate', metavar='R', default=16384.0, type=float, help='sample rate (in Hz)')

    parser.add_argument('--fmin', metavar='F', default=20.0, type=float, help='minimum frequency for integration/waveform')
    
    parser.add_argument('--malmquist-snr', metavar='SNR', type=float, help='SNR threshold for Malmquist prior')

    parser.add_argument('--mmin', metavar='M', default=1.0, type=float, help='minimum component mass')
    parser.add_argument('--mmax', metavar='M', default=35.0, type=float, help='maximum component mass')
    parser.add_argument('--dmax', metavar='D', default=1000.0, type=float, help='maximim distance (Mpc)')

    parser.add_argument('--inj-params', metavar='FILE', help='file containing injection parameters')
    parser.add_argument('--inj-xml', metavar='XML_FILE', help='LAL XML containing sim_inspiral table')
    parser.add_argument('--event', metavar='N', type=int, default=0, help='row index in XML table')

    parser.add_argument('--start-position', metavar='FILE', help='file containing starting positions for chains')

    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of ensemble walkers')
    parser.add_argument('--nensembles', metavar='N', type=int, default=100, help='number of ensembles to accumulate')
    parser.add_argument('--nthin', metavar='N', type=int, default=10, help='number of setps to take between each saved ensemble state')

    parser.add_argument('--nthreads', metavar='N', type=int, default=1, help='number of concurrent threads to use')

    parser.add_argument('--Tmax', metavar='T', type=float, default=133.0, help='maximum temperature in the PT ladder')
    parser.add_argument('--Tstep', metavar='dT', type=float, help='ratio between successive temperatures')

    parser.add_argument('--restart', default=False, action='store_true', help='continue a previously-existing run')

    args=parser.parse_args()

    if len(args.ifo) == 0:
        args.ifo = ['H1', 'L1', 'V1']

    time_data = None
    if args.data is not None:
        time_data = list(np.transpose(np.loadtxt(args.data)))

    # By default, start at GPS 0
    gps_start = lal.LIGOTimeGPS(0)
    if args.data_start_sec is not None:
        gps_start.gpsSeconds = args.data_start_sec
        if args.data_start_ns is not None:
            gps_start.gpsNanoSeconds = args.data_start_ns

    inj_params = None
    if args.inj_params is not None:
        inj_params = np.loadtxt(args.inj_params)
    elif args.inj_xml is not None:
        inj_params = params.inj_xml_to_params(args.inj_xml, event=args.event, time_offset=gps_start)

    lnposterior = \
            pos.TimeMarginalizedPosterior(time_data=time_data,
                                          inj_params=inj_params,
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
                                          detectors=args.ifo)

    if args.Tstep is None:
        ndim = params.nparams
        idim = ndim + 1

        if idim >= len(t_steps):
            tstep = 1 + np.sqrt(2.0/float(ndim))
        else:
            tstep = t_steps[idim]
    else:
        tstep = args.Tstep
        
    Ts = np.exp(np.arange(0.0, np.log(args.Tmax), np.log(tstep)))
    NTs = Ts.shape[0]    

    # Set up initial configuration
    nparams = params.nparams_time_marginalized
    p0 = np.zeros((NTs, args.nwalkers, nparams))
    means = []
    if args.restart:
        for i in range(NTs):
            p0[i, :, :] = np.loadtxt('chain.%02d.dat'%i)[-args.nwalkers:,:]

        means = list(np.mean(np.loadtxt('chain.00.dat').reshape((-1, args.nwalkers, nparams)), axis=1))
    elif args.start_position is not None:
        p0 = np.loadtxt(args.start_position).reshape((NTs, args.nwalkers, nparams))
        
        if args.malmquist_snr is not None:
            fix_malmquist(p0, lnposterior, args.malmquist_snr, nthreads=args.nthreads)
    else:
        for i in range(NTs):
            p0[i,:,:] = lnposterior.draw_prior(shape=(args.nwalkers,)).view(float).reshape((args.nwalkers, nparams))

        if args.malmquist_snr is not None:
            fix_malmquist(p0, lnposterior, args.malmquist_snr, nthreads=args.nthreads)

    sampler = emcee.PTSampler(NTs, args.nwalkers, nparams, LogLikelihood(lnposterior), 
                              LogPrior(lnposterior), threads = args.nthreads, 
                              betas = 1.0/Ts)

    np.savetxt('temperatures.dat', Ts.reshape((1,-1)))
    with open('sampler-params.dat', 'w') as out:
        out.write('# NTemps NWalkers Nthin\n')
        out.write('{0:d} {1:d} {2:d}\n'.format(NTs, args.nwalkers, args.nthin))

    freq_data_columns = (lnposterior.fs,)
    for d in lnposterior.data:
        freq_data_columns = freq_data_columns + (np.real(d), np.imag(d))
    np.savetxt('freq-data.dat', np.column_stack(freq_data_columns))

    with open('command-line.txt', 'w') as out:
        out.write(' '.join(sys.argv) + '\n')

    # Set up headers:
    if not args.restart:
        for i in range(NTs):
            with open('chain.%02d.dat'%i, 'w') as out:
                header = ' '.join(map(lambda (n,t): n, params.params_time_marginalized_dtype))
                out.write('# ' + header + '\n')

            with open('chain.%02d.lnlike.dat'%i, 'w') as out:
                out.write('# lnlike0 lnlike1 ...\n')

            with open('chain.%02d.lnpost.dat'%i, 'w') as out:
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

                continue

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

            print 'Found new best likelihood of {0:5g}.'.format(old_best_lnlike)
            print 'Resetting around parameters '
            print '\n'.join(['{0:<15s}: {1:>15.8g}'.format(n, v) for ((n, t), v) in zip(params.params_time_marginalized_dtype, best)])
            print 
            sys.stdout.flush()

            # Iterate one more time before storing the new parameters
            continue

        if reset:
            reset_files(NTs)
            reset = False
        for i in range(NTs):
            with open('chain.{0:02d}.dat'.format(i), 'a') as out:
                np.savetxt(out, p0[i,:,:])
            with open('chain.{0:02d}.lnlike.dat'.format(i), 'a') as out:
                np.savetxt(out, lnlike[i,:].reshape((1,-1)))
            with open('chain.{0:02d}.lnpost.dat'.format(i), 'a') as out:
                np.savetxt(out, lnpost[i,:].reshape((1,-1)))

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
