import lal
import numpy as np
from pylal import SimInspiralUtils

nparams = 15
params_dtype = [('log_mc', np.float),
                ('eta', np.float),
                ('cos_iota', np.float),
                ('phi', np.float),
                ('psi', np.float),
                ('time', np.float),
                ('ra', np.float),
                ('sin_dec', np.float),
                ('log_dist', np.float),                    
                ('a1', np.float),
                ('cos_tilt1', np.float),
                ('phi1', np.float),
                ('a2', np.float),
                ('cos_tilt2', np.float),
                ('phi2', np.float)]
params_latex = [r'\log \mathcal{M}',
                r'\eta',
                r'\cos(\iota)',
                r'\phi',
                r'\psi',
                r't',
                r'\alpha',
                r'\sin(\delta)',
                r'\log d',
                r'a_1',
                r'\cos\left(t_1\right)',
                r'\phi_1',
                r'a_2',
                r'\cos\left(t_2\right)',
                r'\phi_2']

def to_params(arr):
    """Returns a view of ``arr`` with labeled columns.  See
    :data:`params_dtype` for column names.
    """
    return arr.view(np.dtype(params_dtype))

def from_params(arr):
    """Returns a view of ``arr`` as a float array consistent with
    :data:`params_dtype`"""
    shape = arr.shape

    return arr.view(float).reshape(shape+(-1,))

nparams_time_marginalized = 14
params_time_marginalized_dtype = [('log_mc', np.float),
                                  ('eta', np.float),
                                  ('cos_iota', np.float),
                                  ('phi', np.float),
                                  ('psi', np.float),
                                  ('ra', np.float),
                                  ('sin_dec', np.float),
                                  ('log_dist', np.float),
                                  ('a1', np.float),
                                  ('cos_tilt1', np.float),
                                  ('phi1', np.float),
                                  ('a2', np.float),
                                  ('cos_tilt2', np.float),
                                  ('phi2', np.float)]
params_time_marginalized_latex = [r'\log \mathcal{M}',
                                  r'\eta',
                                  r'\cos(\iota)',
                                  r'\phi',
                                  r'\psi',
                                  r'\alpha',
                                  r'\sin(\delta)',
                                  r'\log d',
                                  r'a_1',
                                  r'\cos\left(t_1\right)',
                                  r'\phi_1',
                                  r'a_2',
                                  r'\cos\left(t_2\right)',
                                  r'\phi_2']

def to_time_marginalized_params(arr):
    """Returns a view of ``arr`` with labeled columns."""
    return arr.view(params_time_marginalized_dtype)
    
def from_time_marginalized_params(arr):
    """Returns a view of ``arr`` as a float array, assuming it is of
    ``params_time_marginalized_dtype``."""

    shape = arr.shape

    return arr.view(float).reshape(shape + (-1,))

def params_to_time_marginalized_params(arr):
    arr = to_params(arr)
    shape = arr.shape

    arr_tm = np.zeros(shape, dtype=params_time_marginalized_dtype)

    for n,t in params_time_marginalized_dtype:
        arr_tm[n] = arr[n]

    return arr_tm

def time_marginalized_params_to_params(arr, time=0):
    """Returns an array of params corresponding to ``arr`` with the
    ``time`` column added and assigned the value in ``time``.
    """
    arr = to_time_marginalized_params(arr)

    arr_p = to_params(np.zeros(arr.shape, dtype=params_dtype))

    for n in arr.dtype.names:
        arr_p[n] = arr[n]

    arr_p['time'] = time

    return arr_p

def inj_xml_to_params(inj, event=0, time_offset=lal.LIGOTimeGPS(0)):
    """Turns a LAL sim_inspiral table from an XML file into parameters for
    a waveform.

    :param inj: Filename of the XML file.

    :param event: Row index of parameters in table.

    :time_offset: GPS time of the start of the data segment being
      analyized.

    """

    table = SimInspiralUtils.ReadSimInspiralFromFiles([inj])[event]

    p = np.zeros(1, dtype=params_dtype)

    p['log_mc'] = np.log(table.mchirp)
    p['eta'] = table.eta
    p['log_dist'] = np.log(table.distance)
    p['ra'] = table.longitude
    p['sin_dec'] = np.sin(table.latitude)
    p['cos_iota'] = np.cos(table.inclination)
    p['phi'] = table.coa_phase
    p['psi'] = table.polarization
    
    p['time'] = table.geocent_end_time - time_offset.gpsSeconds + 1e-9*(table.geocent_end_time_ns - time_offset.gpsNanoSeconds)

    s1 = np.array([table.spin1x, table.spin1y, table.spin1z])
    s2 = np.array([table.spin2x, table.spin2y, table.spin2z])

    Lhat = np.array([np.sin(table.inclination), 0.0, np.cos(table.inclination)])
    xhat = np.array([np.cos(table.inclination), 0.0, -np.sin(table.inclination)])
    yhat = np.array([0.0,1.0,0.0])

    if np.linalg.norm(s1) == 0.0:
        p['a1'] = 0.0
        p['cos_tilt1'] = 1.0
        p['phi1'] = 0.0
    else:
        a1 = np.linalg.norm(s1)
        p['a1'] = a1
        p['cos_tilt1'] = np.dot(s1, Lhat)/a1
        p['phi1'] = np.arctan2(np.dot(s1, yhat), np.dot(s1, xhat))

    if np.linalg.norm(s2) == 0.0:
        p['a2'] = 0.0
        p['cos_tilt2'] = 1.0
        p['phi2'] = 0.0
    else:
        a2 = np.linalg.norm(s2)
        p['a2'] = a2
        p['cos_tilt2'] = np.dot(s2, Lhat)/a2
        p['phi2'] = np.arctan2(np.dot(s2, yhat), np.dot(s2, xhat))

    return p
        
def fix_spherical(p, sin_lat_name, long_name):
    """Fixes the given latitude and longitude coordinates in ``p`` to
    their canonical ranges.  Going 'over the poles' is handled
    properly.  ``p`` is altered by this function.

    """

    p[long_name] = np.fmod(p[long_name], 2.0*np.pi)
    if p[long_name] < 0.0:
        p[long_name] += 2.0*np.pi

    p[sin_lat_name] = np.fmod(p[sin_lat_name], 2.0)
    if p[sin_lat_name] > 1.0:
        p[sin_lat_name] = 2.0-p[sin_lat_name]
        p[long_name] = np.fmod(p[long_name] + np.pi, 2.0*np.pi)
    if p[sin_lat_name] < -1.0:
        p[sin_lat_name] = -2.0 - p[sin_lat_name]
        p[long_name] = np.fmod(p[long_name] + np.pi, 2.0*np.pi)

def normalize_coordinates(p):
    """Returns a set of parameters that will produce an identical waveform
    to ``p``, but with values in canonical ranges. ``p`` should
    already be in parameter form (i.e. with appropriate column
    names).

    """

    p = p.copy()

    p['phi'] = np.fmod(p['phi'], 2.0*np.pi)
    if p['phi'] < 0.0:
        p['phi'] += 2.0*np.pi

    fix_spherical(p, 'cos_iota', 'psi')
    fix_spherical(p, 'sin_dec', 'ra')
    fix_spherical(p, 'cos_tilt1', 'phi1')
    fix_spherical(p, 'cos_tilt2', 'phi2')

    return p
