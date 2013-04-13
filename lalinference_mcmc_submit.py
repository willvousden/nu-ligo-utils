#!/usr/bin/env python
import os
import argparse
import getpass
import socket
import numpy as np

import lalsimulation as lalsim
from glue.ligolw import lsctables
from glue.ligolw import ligolw
from pylal import SimInspiralUtils

from lalsim_snr import *


def exists(filename):
    """Check if filename exists"""
    try:
        with open(filename): return True
    except IOError:
        print "Warning: {} not found.".format(rc)
        return False


parser = argparse.ArgumentParser(description='Generate a submit file for lalinference_mcmc on grail.')

msub = parser.add_argument_group('MSUB')
env = parser.add_argument_group('env')
li_mcmc = parser.add_argument_group('lalinference_mcmc')

msub.add_argument('--alloc', default='b1011',
        help='Allocation to charge SUs to (default=b1011).')
msub.add_argument('--queue', default='buyin',
        help='Queue for job (default=buyin).')
msub.add_argument('--jobName',
        help='Name of job, used for output file names and queue listing.')
msub.add_argument('--dir', default=os.getcwd(),
        help='Directory where submit script is written and executed from (default=current directory).')
msub.add_argument('--dep', default=None,
        help='Wait for dependent job ID to complete (default=None).')
msub.add_argument('--name', default='submit',
        help='Name of submit file (default=submit).')
msub.add_argument('--walltime', default='2:00:00:00',
        help='Walltime for job (default=2:00:00:00).')
msub.add_argument('--nPar', default=None,
        help='Number of dimensions for MCMC.  Defaults for common templates are set, assuming no fixed parameters or PSD fitting.')

env.add_argument('--branch', default='master',
        help='Branchname to use, assuming /projects/p20251/USER/lsc/BRANCHNAME/etc/lscsoftrc exists (default=master).')
env.add_argument('--rc', action='append',
        help='Specify direct path to rc files to be sourced (e.g. lscsoftrc).  /projects/p20128/non-lsc/lscsoft-user-env.sh is added by default.')
li_mcmc.add_argument('--era', required=True, default='advanced',
        help='Era ("initial" or "advanced") of detector PSD for SNR calculation.  If no cache arguments given, this will add the appropriate analytical PSD arguments to the submit file.')
li_mcmc.add_argument('--ifo', default=['H1','L1','V1'], action='append',
        help='IFOs for the analysis.')
li_mcmc.add_argument('--inj',
        help='Injection XML file.')
li_mcmc.add_argument('--event', type=int,
        help='Event number in XML to inject.')
li_mcmc.add_argument('--approx', default=None,
        help='Specify a template approximant (default=max available for APPROX).')
li_mcmc.add_argument('--ampOrder', default=None,
        help='Specify amplitude order of template.')
li_mcmc.add_argument('--flow', default=40., type=float,
        help='Lower frequency bound for all detectors (default=40).')
li_mcmc.add_argument('--srate', default=None, type=float,
        help='Sampling rate of the waveform.  If not provided and an injection is peformed, it is set to be sufficient for the signal being injected.  If no injection, it defaults to a sufficient value for a 1.4-1.4 binary coalescence (expensive!).')
li_mcmc.add_argument('--seglen', default=None, type=float,
        help='Length of data segment used for likelihood compuatation.  Same default behavior as "--srate".')
li_mcmc.add_argument('--tempLadderBottomUp', default=False, action='store_true',
        help='Build the temperature ladder from the bottom up, using an analytic prescription for the spacing that should ensure communication between chains.  Sets the number of cores so that the hottest temperature should be sampling the prior.')
li_mcmc.add_argument('--trigSNR', default=None, type=float,
        help='SNR of the trigger (calculated automatically if injecting).')
li_mcmc.add_argument('--tempMin', default=1.0, type=float,
        help='Temperature of coldest chain (default=1.0).')
li_mcmc.add_argument('--tempMax', type=float,
        help='Temperature of hotest chain.  Determined automatically if injecting, or trigSNR is given.')

args,unknown = parser.parse_known_args()

# Assume all unknown arguments are meant for lalinference_mcmc
li_args = unknown

# Directories to look for lscsoftrc files on Quest
user_dict = {'bff394':'bfarr',
            'wem989':'w-farr',
            'tbl987':'tyson'}

# If not on quser##, don't bother making a submit file
if 'quser' in socket.gethostname():
    on_quest = True
else:
    on_quest = False

# Analytic PSDs.
# FIXME: The SNR calculated uses lalsimulation PSDs.  LALInference uses lalinspiral
# which doesn't include all detectors in the advanced era.  Thus the SNRs
# calculated here may not match what LALInference reports.  LALInference should be
# updated to use the lalsimulation functions.
noise_psd_caches = {}
print "Using {}-era PSDs.".format(args.era)
if args.era == 'initial':
    for _ifos, _cache in (
      (('H1', 'H2', 'L1', 'I1'), 'LALLIGO'),
      (('V1',), 'LALVirgo')):
        for _ifo in _ifos:
            noise_psd_caches[_ifo] = _cache

elif args.era == 'advanced':
    for _ifos, _cache in (
      (('H1', 'H2', 'L1', 'I1', 'V1'), 'LALAdLIGO'),):
        for _ifo in _ifos:
            noise_psd_caches[_ifo] = _cache


# Check if caches specified in extra arguments
cache_check = ['-cache' in arg for arg in li_args]
caches_specified = True if True in cache_check else False

# Default number of parameters for a few common templates.  Assume no fixed
#  parameters and no PSD fitting
nPars = {lalsim.SpinTaylorT4:15,
        lalsim.IMRPhenomB:11,
        lalsim.TaylorF2:9, lalsim.TaylorF2RedSpin:11,
        lalsim.TaylorT1:9, lalsim.TaylorT2:9, lalsim.TaylorT3:9, lalsim.TaylorT4:9,
        lalsim.EOB:9, lalsim.EOBNR:9, lalsim.EOBNRv2:9, lalsim.EOBNRv2HM:9,
        lalsim.SEOBNRv1:11}

# Target likelihood value "sampled" by the hottest chain.  Same as in lalinference_mcmc.
target_hot_like = 15.

# Maximum sampling rate
srate_max = 16384

submitFilePath = os.path.join(args.dir, args.name)

# Setup and check envirnment files to be sourced
rcs = args.rc if args.rc else []

# Add non-lsc standard location if one is not given
non_lsc_check = ['non-lsc' in rc_path for rc_path in rcs]
if True not in non_lsc_check and on_quest:
    rcs.append('/projects/p20128/non-lsc/lscsoft-user-env.sh')

# Assume a certain directory tree structure if branch name is given
if args.branch and on_quest:
    try:
        lscsoftrc = '/projects/p20251/{}/lsc/{}/etc/lscsoftrc'.format(user_dict[getpass.getuser()],args.branch)
    except KeyError:
        lscsoftrc = '/projects/p20251/{}/lsc/{}/etc/lscsoftrc'.format(getpass.getuser(),args.branch)

    rcs.append(lscsoftrc)

# Only include rc files that exist
rcs[:] = [rc for rc in rcs if exists(rc)]

# Necessary modules
modules = ['python','mpi/openmpi-1.6.3-intel2013.2']

# Determine sampling rate, segment length, and SNR (--trigSNR takes precedence).
if args.ampOrder is None:
    amp_order = None
else:
    try:
        amp_order = int(args.ampOrder)
    except ValueError:
        amp_order = lalsim.GetOrderFromString(args.ampOrder)

SNR = None
if args.trigSNR:
    calcSNR=False
else:
    calcSNR = True


if args.inj and args.event is not None:
    SNR, srate, seglen, flow = get_inj_info(amp_order, args.inj, args.event, args.ifo, args.era, args.flow, calcSNR)
else:
    print "No injections, using BNS as a conservative reference."
    srate, seglen, flow = get_bns_info(args.flow)

if args.trigSNR:
    SNR = args.trigSNR

if args.srate:
    srate = args.srate

if args.seglen:
    seglen = args.seglen

if srate > srate_max:
    srate = srate_max


# Ladder spacing flat in the log.  Analytic delta
if args.tempLadderBottomUp:
    li_args.append('--tempLadderBottomUp')

    # Determine maximum temperature
    temp_max = None
    if args.tempMax:
        temp_max = args.tempMax

    elif SNR is not None:
        max_log_l = SNR*SNR/2
        temp_max = max_log_l / target_hot_like
    print "Max temperature of {} needed.".format(temp_max)

    # Determine spacing
    if args.nPar:
        nPar = args.nPar
    else:
        if args.approx:
            approx = lalsim.GetApproximantFromString(args.approx)
        else:
            print "No approximant or number of parameters given. Defaulting to conservative temperature spacing, with nPar=15."
            approx = lalsim.SpinTaylorT4

        try:
            nPar = nPars[approx]
        except KeyError:
            raise(KeyError, "No default value for given approx.  Specity explicitly with the --nPar argument.")

    temp_delta = 1 + np.sqrt(2./nPar)

    n_chains = 1
    temp_min = args.tempMin
    temp = temp_min
    while temp < temp_max:
        n_chains += 1
        temp = temp_min * np.power(temp_delta, n_chains)
    print "{} n_chain needed.".format(n_chains)

# Prepare lalinference_mcmc arguments
ifos = args.ifo
ifo_args = ['--ifo {}'.format(ifo) for ifo in ifos]
flow_args = ['--{}-flow {:g}'.format(ifo, flow) for ifo in ifos]

if not caches_specified:
    cache_args = ['--{}-cache {}'.format(ifo, noise_psd_caches[ifo]) for ifo in ifos]

if on_quest:
    with open(submitFilePath,'w') as outfile:
        # MSUB directives
        outfile.write('#!/bin/bash\n')
        outfile.write('#MSUB -A {}\n'.format(args.alloc))
        outfile.write('#MSUB -q {}\n'.format(args.queue))

        outfile.write('#MSUB -l walltime={}\n'.format(args.walltime))
        outfile.write('#MSUB -l nodes=1:ppn={}\n'.format(n_chains))

        if args.dep:
            outfile.write('#MSUB -l {}\n'.format(args.dep))

        if args.jobName:
            outfile.write('#MSUB -N {}\n'.format(args.jobName))

        outfile.write('#MSUB -j oe\n')
        outfile.write('#MOAB -d {}\n'.format(args.dir))
        outfile.write('\n')

        # Ensure core dump on failure
        outfile.write('ulimit -c unlimited\n')

        # Load modules
        outfile.writelines(['module load {}\n'.format(module) for module in modules])
        outfile.write('\n')

        # Load LALSuite environment
        outfile.writelines(['source {}\n'.format(rc) for rc in rcs])
        outfile.write('\n')

        # lalinference_mcmc command line
        outfile.write('mpirun lalinference_mcmc\\\n')
        outfile.write('  {}\\\n'.format(' '.join(ifo_args)))
        outfile.write('  {}\\\n'.format(' '.join(flow_args)))

        if not caches_specified:
            outfile.write('  {}\\\n'.format(' '.join(cache_args)))

        outfile.write('  --srate {:g}\\\n'.format(srate))
        outfile.write('  --seglen {:g}\\\n'.format(seglen))
        outfile.write('  {}'.format(' '.join(li_args)))
