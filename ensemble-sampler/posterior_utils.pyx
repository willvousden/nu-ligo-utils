cdef extern from "math.h":
    double M_PI
    double cos(double x)
    double sin(double x)
    double log1p(double x)
    double exp(double x)
    double log(double x)
    double sqrt(double x)
    
cdef extern from "complex.h":
    double complex cexp(double complex x)
    double complex I
    double creal(double complex x)
    double cimag(double complex x)
    double complex conj(double complex x)

cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
def combine_and_timeshift(double fplus,
                          double fcross,
                          np.ndarray[np.complex128_t, ndim=1] hplus,
                          np.ndarray[np.complex128_t, ndim=1] hcross,
                          np.ndarray[np.float_t, ndim=1] fs,
                          double timeshift):
    cdef unsigned int NFull = fs.shape[0]
    cdef unsigned int Nh = hplus.shape[0]
    
    assert hcross.shape[0] == Nh, 'hplus and hcross must have the same shape'

    cdef unsigned int N = min(NFull, Nh)

    cdef unsigned int i

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

@cython.boundscheck(False)
def data_waveform_inner_product(unsigned int istart,
                                double df,
                                np.ndarray[np.float64_t, ndim=1] psd,
                                np.ndarray[np.complex128_t, ndim=1] h,
                                np.ndarray[np.complex128_t, ndim=1] d):
    cdef unsigned int N = psd.shape[0]
    cdef unsigned int i
    cdef double hh = 0.0
    cdef double dh = 0.0
    cdef double dd = 0.0

    assert h.shape[0] == N, 'shape of psd and waveform must match'
    assert d.shape[0] == N, 'shape of psd, waveform, and data must match'
    assert istart < N, 'istart must be smaller than data length'

    
    for i in range(istart, N):
        hh += 4.0*df*creal(conj(h[i])*h[i])/psd[i]
        dh += 4.0*df*creal(conj(d[i])*h[i])/psd[i]
        dd += 4.0*df*creal(conj(d[i])*d[i])/psd[i]

    return hh,dh,dd

@cython.boundscheck(False)
def logaddexp_sum(np.ndarray[np.float64_t, ndim=1] arr):
    cdef unsigned int N = arr.shape[0]
    cdef unsigned int i
    cdef double log_sum = arr[0]
    cdef double log_term

    for i in range(1,N):
        log_term = arr[i]

        if log_sum > log_term:
            log_sum += log1p(exp(log_term-log_sum))
        else:
            log_sum = log_term + log1p(exp(log_sum-log_term))

    return log_sum
            
@cython.boundscheck(False)
def logaddexp_sum_bessel(np.ndarray[np.float64_t, ndim=1] bessel_scaled,
                         np.ndarray[np.float64_t, ndim=1] scaling):
    cdef unsigned int N = bessel_scaled.shape[0]
    cdef unsigned int i
    cdef double log_sum
    cdef double log_term

    assert scaling.shape[0] == N, 'arrays must be same shapes'

    log_sum = log(bessel_scaled[0]) + scaling[0]

    for i in range(1,N):
        log_term = log(bessel_scaled[i]) + scaling[i]

        if log_sum > log_term:
            log_sum += log1p(exp(log_term - log_sum))
        else:
            log_sum = log_term + log1p(exp(log_sum - log_term))

    return log_sum

@cython.boundscheck(False)
def hh_dd_sum(double df,
              np.ndarray[np.float64_t, ndim=1] psd,
              np.ndarray[np.complex128_t, ndim=1] h, 
              np.ndarray[np.complex128_t, ndim=1] d):
    cdef unsigned int N = psd.shape[0]
    cdef unsigned int i
    cdef double sh = 0.0
    cdef double sd = 0.0
    cdef double re
    cdef double im

    assert h.shape[0] == N and d.shape[0] == N, 'array shapes must match'

    for i in range(N):
        re = creal(h[i])
        im = cimag(h[i])
        sh += 4.0*df*(re*re + im*im)/psd[i]
        re = creal(d[i])
        im = cimag(d[i])
        sd += 4.0*df*(re*re + im*im)/psd[i]

    return sh, sd

@cython.boundscheck(False)
def fill_fft_array(double df, 
                   np.ndarray[np.float64_t, ndim=1] psd,
                   np.ndarray[np.complex128_t, ndim=1] d,
                   np.ndarray[np.complex128_t, ndim=1] h,
                   np.ndarray[np.complex128_t, ndim=1] fft_array):
    cdef unsigned int N = psd.shape[0]
    cdef unsigned int i
    cdef double re 
    cdef double im

    assert (d.shape[0] == N) and (h.shape[0] == N) and (fft_array.shape[0] == N), 'array shapes must match'

    for i in range(N):
        re = 2.0*df*(creal(d[i])*creal(h[i]) + cimag(d[i])*cimag(h[i]))/psd[i]
        im = 2.0*df*(-cimag(d[i])*creal(h[i]) + creal(d[i])*cimag(h[i]))/psd[i]
        fft_array[i] = re + im*1j

@cython.boundscheck(False)
def fill_fft_array_real(double df, 
                        np.ndarray[np.float64_t, ndim=1] psd,
                        np.ndarray[np.complex128_t, ndim=1] d,
                        np.ndarray[np.complex128_t, ndim=1] h,
                        np.ndarray[np.complex128_t, ndim=1] fft_array):
    cdef unsigned int N = psd.shape[0]
    cdef unsigned int i
    cdef double re
    cdef double im

    assert (d.shape[0] == N) and (h.shape[0] == N) and (fft_array.shape[0] == N), 'array shapes must match'

    for i in range(N):
        re = 2.0*df*creal(d[i])*creal(h[i])/psd[i]
        im = -2.0*df*cimag(d[i])*creal(h[i])/psd[i]
        fft_array[i] = re + im*1j

@cython.boundscheck(False)
def fill_fft_array_imag(double df, 
                        np.ndarray[np.float64_t, ndim=1] psd,
                        np.ndarray[np.complex128_t, ndim=1] d,
                        np.ndarray[np.complex128_t, ndim=1] h,
                        np.ndarray[np.complex128_t, ndim=1] fft_array):
    cdef unsigned int N = psd.shape[0]
    cdef unsigned int i
    cdef double re
    cdef double im

    assert (d.shape[0] == N) and (h.shape[0] == N) and (fft_array.shape[0] == N), 'array shapes must match'

    for i in range(N):
        re = 2.0*df*creal(d[i])*cimag(h[i])/psd[i]
        im = -2.0*df*cimag(d[i])*cimag(h[i])/psd[i]
        fft_array[i] = re + im*1j

@cython.boundscheck(False)
def twice_norm(np.ndarray[np.float64_t, ndim=1] real_part,
               np.ndarray[np.float64_t, ndim=1] imag_part,
               np.ndarray[np.float64_t, ndim=1] norm2):
    cdef unsigned int N = real_part.shape[0]
    cdef unsigned int i
    cdef double re
    cdef double im

    assert imag_part.shape[0] == N, 'real_part and imag_part shapes must match'
    assert norm2.shape[0] == N, 'real_part and storage shapes must match'

    for i in range(N):
        re = real_part[i]
        im = imag_part[i]
        norm2[i] = 2.0*sqrt(re*re + im*im)
