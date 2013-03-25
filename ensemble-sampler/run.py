#!/usr/bin/env python

from argparse import ArgumentParser
import emcee
import lal
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

if __name__ == '__main__':
    parser = ArgumentParser(description='run an ensemble MCMC analysis of a GW event')

    parser.add_argument('--dataseed', metavar='N', type=int, help='seed for data generation')

    parser.add_argument('--data', metavar='FILE', help='file containing time-domain strain data for analysis')
    parser.add_argument('--data-start-sec', metavar='N', type=int, help='GPS integer seconds of data start')
    parser.add_argument('--data-start-ns', metavar='N', type=int, help='GPS nano-seconds of data start')

    parser.add_argument('--seglen', metavar='DT', type=float, help='data segment length')

    parser.add_argument('--srate', metavar='R', default=16384.0, type=float, help='sample rate (in Hz)')
    
    parser.add_argument('--malmquist-snr', metavar='SNR', type=float, help='SNR threshold for Malmquist prior')

    parser.add_argument('--mmin', metavar='M', default=1.0, type=float, help='minimum component mass')
    parser.add_argument('--mmax', metavar='M', default=35.0, type=float, help='maximum component mass')
    parser.add_argument('--dmax', metavar='D', default=1000.0, type=float, help='maximim distance (Mpc)')

    parser.add_argument('--inj-params', metavar='FILE', help='file containing injection parameters')
    parser.add_argument('--start-params', metavar='FILE', help='file containing a good initial location in parameter space')

    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of ensemble walkers')
    parser.add_argument('--nensembles', metavar='N', type=int, default=100, help='number of ensembles to accumulate')
    parser.add_argument('--nthin', metavar='N', type=int, default=1000, help='number of setps to take between each saved ensemble state')
    parser.add_argument('--nburnin', metavar='N', type=int, help='number of steps to discard as burnin')
    parser.add_argument('--nthreads', metavar='N', type=int, default=1, help='number of concurrent threads to use')

    parser.add_argument('--Tmax', metavar='T', type=float, default=133.0, help='maximum temperature in the PT ladder')
    parser.add_argument('--Tstep', metavar='dT', type=float, help='ratio between successive temperatures')

    parser.add_argument('--restart', default=False, action='store_true', help='continue a previously-existing run')

    args=parser.parse_args()

    time_data = None
    if args.data is not None:
        time_data = np.loadtxt(args.data)

    # By default, start at GPS 0
    gps_start = lal.LIGOTimeGPS(0)
    if args.data_start_sec is not None:
        gps_start.gpsSeconds = args.data_start_sec
        if args.data_start_ns is not None:
            gps_start.gpsNanoSeconds = args.data_start_ns

    inj_params = None
    if args.inj_params is not None:
        inj_params = np.loadtxt(args.inj_params)

    lnpost = pos.Posterior(time_data=time_data, inj_params=inj_params, T=args.seglen, 
                           time_offset=gps_start, srate=args.srate, malmquist_snr=args.malmquist_snr, 
                           mmin=args.mmin, mmax=args.mmax, dmax=args.dmax, dataseed=args.dataseed)

    if args.restart:
        nburnin = 0
    elif args.nburnin is None:
        nburnin = int(round(0.1*(args.nensembles*args.nthin)))
    else:
        nburnin = args.nburnin

    if args.Tstep is None:
        tstep = 1 + np.sqrt(2.0/float(pos.nparams))
    else:
        tstep = args.Tstep
        
    Ts = np.exp(np.arange(0.0, np.log(args.Tmax), np.log(tstep)))
    NTs = Ts.shape[0]    

    # Set up initial configuration
    p0 = np.zeros((NTs, args.nwalkers, pos.nparams))
    sigma = pos.params_sigma.view((float, pos.nparams)).reshape((pos.nparams,))
    if args.restart:
        for i in range(NTs):
            p0[i, :, :] = np.loadtxt('chain.%02d.dat'%i)[-args.nwalkers:,:]
    elif args.start_params is not None:
        params0 = np.loadtxt(args.start_params)
        for i in range(NTs):
            p0[i,:,:] = pos.sample_ball(params0, args.nwalkers).view(float).reshape((args.nwalkers, pos.nparams))
    elif args.inj_params is not None:
        params0 = np.loadtxt(args.inj_params)
        for i in range(NTs):
            p0[i,:,:] = pos.sample_ball(params0, args.nwalkers).view(float).reshape((args.nwalkers, pos.nparams))
    else:
        raise ValueError('must specify one of \'restart\', \'start-params\', or \'inj-params\' for starting position')

    sampler = emcee.PTSampler(NTs, args.nwalkers, pos.nparams, LogLikelihood(lnpost), 
                              LogPrior(lnpost), threads = args.nthreads, 
                              betas = 1.0/Ts)

    np.savetxt('temperatures.dat', Ts.reshape((1,-1)))
    with open('sampler-params.dat', 'w') as out:
        out.write('# NTemps NWalkers\n')
        out.write('%d %d\n'%(NTs, args.nwalkers))

    freq_data_columns = (lnpost.fs,)
    for d in lnpost.data:
        freq_data_columns = freq_data_columns + (np.real(d), np.imag(d))
    np.savetxt('freq-data.dat', np.column_stack(freq_data_columns))

    with open('command-line.txt', 'w') as out:
        out.write(' '.join(sys.argv) + '\n')

    print 'Beginning burnin.'
    sys.stdout.flush()

    for p0, lnprob, lnlike in sampler.sample(p0, iterations=nburnin, storechain=False):
        pass
    print 'afrac: ', np.mean(sampler.acceptance_fraction, axis=1)
    print 'tfrac: ', sampler.tswap_acceptance_fraction
    print '\n\n'
    sys.stdout.flush()
    sampler.reset()

    # Set up headers:
    if not args.restart:
        for i in range(NTs):
            with open('chain.%02d.dat'%i, 'w') as out:
                header = ' '.join(map(lambda (n,t): n, pos.params_dtype))
                out.write('# ' + header + '\n')

            with open('chain.%02d.lnlike.dat'%i, 'w') as out:
                out.write('# lnlike0 lnlike1 ...\n')

            with open('chain.%02d.lnpost.dat'%i, 'w') as out:
                out.write('# lnpost0 lnpost1 ...\n')

    print 'Beginning run'
    sys.stdout.flush()
    for i, (p0, lnprob, lnlike) in enumerate(sampler.sample(p0, iterations=args.nensembles*args.nthin, 
                                                            thin=args.nthin, storechain=False)):
        if (i+1)%args.nthin == 0:
            for i in range(NTs):
                with open('chain.%02d.dat'%i, 'a') as out:
                    np.savetxt(out, p0[i,:,:])
                with open('chain.%02d.lnlike.dat'%i, 'a') as out:
                    np.savetxt(out, lnlike[i,:].reshape((1,-1)))
                with open('chain.%02d.lnpost.dat'%i, 'a') as out:
                    np.savetxt(out, lnprob[i,:].reshape((1,-1)))

            print 'afrac: ', np.mean(sampler.acceptance_fraction, axis=1)
            print 'tfrac: ', sampler.tswap_acceptance_fraction
            print '\n'
            sys.stdout.flush()
