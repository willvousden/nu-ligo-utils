from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

ext_modules = [Extension('posterior_utils', ['ensemble-sampler/posterior_utils.pyx'], include_dirs=[get_include()])]

setup(
    name = 'NU LIGO utils',
    cmdclass = {'build_ext' : build_ext},
    ext_modules = ext_modules
)
