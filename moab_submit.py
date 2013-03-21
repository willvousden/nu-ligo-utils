#!/usr/bin/env python

import os
import argparse
import getpass
import numpy as np

import lalsimulation as lalsim
from glue.ligolw import lsctables
from glue.ligolw import ligolw
from pylal import SimInspiralUtils

from sim_inspiral_snr import get_inj_info

parser = argparse.ArgumentParser(description='Generate a submit file for lalinference_mcmc on grail.')

moab = parser.add_argument_group('MOAB')
li_mcmc = parser.add_argument_group('lalinference_mcmc')

moab.add_argument('--alloc', default='b1011', help='Allocation for job')
moab.add_argument('--queue', default='buyin', help='Queue for job')
moab.add_argument('--jobName', help='Name of job.')
moab.add_argument('--dir', default=os.getcwd(), help='Directory to write submit script and work from.')
moab.add_argument('--dep', help='Job ID of dependency (if any).')
moab.add_argument('--name', default='submit', help='Name of submit file.')
moab.add_argument('--walltime', default='2:00:00:00', help='Walltime for job.')

li_mcmc.add_argument('--branch', default='master',
  help='Branchname to use (assumes /projects/p20251/USER/lsc/BRANCHNAME/etc/lscsoftrc exists).')
li_mcmc.add_argument('--rc', help="Path to lscsoftrc if branch method doesn't work.")
li_mcmc.add_argument('-i','--inj', help='Injection XML file.')
li_mcmc.add_argument('-e','--event', help='Event number in XML to inject.')
li_mcmc.add_argument('--trigSNR', help='SNR of event.  If not specified and given XML, calculute SNR.')
li_mcmc.add_argument('--era', default='advanced',
    help='Cache file, or analytical PSD to use (for all detectors)')
li_mcmc.add_argument('--flow', default=40, help='Lower frequency bound.')
li_mcmc.add_argument('--ifo', default=['H1','L1','V1'], action='append', help='IFOs for the analysis.')
li_mcmc.add_argument("li_args", nargs=argparse.REMAINDER)

args = parser.parse_args()

user_dict = {'bff394':'bfarr',
            'wem989':'w-farr',
            'tbl987':'tyson'}

nPar = {lalsim.SpinTaylorT4:15,
        lalsim.TaylorF2:9}

submitFilePath = os.path.join(args.dir, args.name)

try:
  lscsoftrc = '/projects/p20251/{}/lsc/{}/etc/lscsoftrc'.format(user_dict[getpass.getuser()],args.branch)
except KeyError:
  lscsoftrc = '/usr/local/etc/{}_rc'.format(args.branch)

modules = ['python','mpi/openmpi-1.6.3-intel2011.3']
rcs = ['/projects/p20128/non-lsc/lscsoft-user-env.sh']
rcs.append(lscsoftrc)

SNR, srate, seglen, flow_hm = get_inj_info(args.inj, args.event, args.ifo, args.era, args.flow)

max_log_l = SNR*SNR/2
temp_delta = 1 + sqrt(2/nPar)

print args.li_args

"""
with open(submitFilePath,'w') as outfile:
  outfile.write('#!/bin/bash\n')
  outfile.write('#MOAB -A {}\n'.format(args.alloc))
  outfile.write('#MOAB -q {}\n'.format(args.queue))
  outfile.write('#MOAB -l walltime={}\n'.format(args.walltime))
  outfile.write('#MOAB -l nodes=1:ppn={}\n'.format())
  outfile.write('\n')

  # Ensure core dump on failure
  outfile.write('ulimit -c unlimited\n')

  # Load modules
  outfile.writelines(['module load {}\n'.format(module) for module in modules])
  outfile.writeline('\n')
"""
