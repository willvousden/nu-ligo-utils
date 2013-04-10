import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double M_PI
    double cos(double x)
    double sin(double x)
    
cdef extern from "complex.h":
    double complex cexp(double complex x)
    double complex I

def combine_and_timeshift(double fplus,
                          double fcross,
                          np.ndarray[np.complex128_t, ndim=1] hplus,
                          np.ndarray[np.complex128_t, ndim=1] hcross,
                          np.ndarray[np.float_t, ndim=1] fs,
                          double timeshift):
    cdef int NFull = fs.shape[0]
    cdef int N = hplus.shape[0]

    cdef int i

    cdef np.ndarray[np.complex128_t, ndim=1] h = np.zeros(NFull, dtype=np.complex128)

    cdef double df = fs[1] - fs[0]
    cdef double sin_pi_df_t = sin(M_PI*df*timeshift)
    cdef double cos_pi_df_t = cos(M_PI*df*timeshift)
    cdef double sin_pi_df_t2 = sin_pi_df_t*sin_pi_df_t
    cdef double complex dshift = -2.0*sin_pi_df_t2 - 2.0*I*cos_pi_df_t*sin_pi_df_t
    cdef double complex shift = 1.0

    for i in range(N):
        hc = hcross[i]
        hp = hplus[i]

        h[i] = shift*(fplus*hplus[i] + fcross*hcross[i])
        shift += shift*dshift

    return h
