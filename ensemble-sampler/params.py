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
