import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double M_PI
    double cos(double x)
    double sin(double x)
    
cdef extern from "complex.h":
    double complex cexp(double complex x)
    double complex I
    double creal(double complex x)
    double cimag(double complex x)
    double complex conj(double complex x)

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

def data_waveform_inner_product(int istart,
                                double df,
                                np.ndarray[np.float64_t, ndim=1] psd,
                                np.ndarray[np.complex128_t, ndim=1] h,
                                np.ndarray[np.complex128_t, ndim=1] d):
    cdef int N = psd.shape[0]
    cdef int i
    cdef double hh = 0.0
    cdef double dh = 0.0

    for i in range(istart, N):
        hh += 4.0*df*creal(conj(h[i])*h[i])/psd[i]
        dh += 4.0*df*creal(conj(d[i])*h[i])/psd[i]

    return hh,dh
