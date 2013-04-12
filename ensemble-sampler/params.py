import numpy as np

nparams = 9
params_dtype = [('log_mc', np.float),
                ('eta', np.float),
                ('cos_iota', np.float),
                ('phi', np.float),
                ('psi', np.float),
                ('time', np.float),
                ('ra', np.float),
                ('sin_dec', np.float),
                ('log_dist', np.float)]

params_latex = [r'\log \mathcal{M}',
                r'\eta',
                r'\cos(\iota)',
                r'\phi',
                r'\psi',
                r't',
                r'\alpha',
                r'\sin(\delta)',
                r'\log d']

def to_params(arr):
    """Returns a view of ``arr`` with labeled columns.  See
    :data:`params_dtype` for column names.
    """
    return arr.view(np.dtype(params_dtype))

def from_params(arr):
    """Returns a view of ``arr`` as a float array consistent with
    :data:`params_dtype`"""
    shape = arr.shape

    return arr.view(float).reshape(shape+(nparams,))

nparams_time_marginalized = 8
params_time_marginalized_dtype = [('log_mc', np.float),
                                  ('eta', np.float),
                                  ('cos_iota', np.float),
                                  ('phi', np.float),
                                  ('psi', np.float),
                                  ('ra', np.float),
                                  ('sin_dec', np.float),
                                  ('log_dist', np.float)]
params_time_marginalized_latex = [r'\log \mathcal{M}',
                                  r'\eta',
                                  r'\cos(\iota)',
                                  r'\phi',
                                  r'\psi',
                                  r'\alpha',
                                  r'\sin(\delta)',
                                  r'\log d']

def to_time_marginalized_params(arr):
    """Returns a view of ``arr`` with labeled columns."""
    return arr.view(params_time_marginalized_dtype)

def from_time_marginalized_params(arr):
    """Returns a view of ``arr`` as a float array, assuming it is of
    ``params_time_marginalized_dtype``."""

    shape = arr.shape

    return arr.view(float).reshape(shape + (nparams_time_marginalized,))

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

    shape = arr.shape

    arr_p = np.zeros(shape, dtype=params_dtype)

    for n,t in params_time_marginalized_dtype:
        arr_p[n] = arr[n]

    arr_p['time'] = time

    return arr_p

    
