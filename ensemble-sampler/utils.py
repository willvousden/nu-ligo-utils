from acor import acor
import lal
import numpy as np

def mc_eta_to_masses(mc, eta):
    r"""Returns masses given chirp mass and symmetric mass ratio.

    :return: ``(m1,m2)``, with :math:`m_1 >= m_2`.
    """
    mtot = mc/eta**0.6
    mprod = eta*mtot*mtot

    disc = mtot*mtot - 4.0*mprod

    sqrt_disc = np.sqrt(disc)

    m1 = 0.5*(mtot + sqrt_disc)
    m2 = 0.5*(mtot - sqrt_disc)

    return (m1,m2)

def masses_to_mc_eta(m1, m2):
    r"""Returns ``(mc,eta)``, chirp mass and symmetric mass ratio given two
    masses.  There is no restriction on which input mass is larger."""
    mtot = m1+m2
    eta = m1*m2/(mtot*mtot)
    mc = mtot*eta**0.6

    return (mc,eta)

class GPSTime(object):
    """Replacement for LIGOTimeGPS, which cannot easily be serialized."""

    def __init__(self, sec=0, ns=0):
        """Initialize with the given seconds and nanoseconds of GPS time."""
        self._sec = sec
        self._ns = ns

    @property
    def sec(self):
        return self._sec

    @property
    def ns(self):
        return self._ns

    @property
    def LIGOTimeGPS(self):
        t = lal.LIGOTimeGPS(0)
        t.gpsSeconds = self.sec
        t.gpsNanoSeconds = self.ns

        return t

def tail(f, n):
    """Returns the last ``n`` lines of the file ``f``."""

    if not (isinstance(f, file)):
        with open(f, 'r') as inp:
            return tail(f, n)

    i = 0
    lines = []
    for l in f:
        if i < n:
            lines.append(l)
            i = i+1
        else:
            lines[i%n] = l
            i = i+1

    if i < n:
        raise ValueError('tail: file must have at least {0:d} lines'.format(n))

    return lines[i%n:] + lines[:i%n]

def thin_chain(chain, fburnin=0.5):
    """Takes a chain of shape ``(Niter, Nwalkers, Nparams)``, and discards
    the first ``fburnin`` of the samples, thinning the remainder by
    the autocorrelation length.

    """

    istart = int(round(fburnin*chain.shape[0]))

    chain = chain[istart:, :, :]

    taumax = float('-inf')
    for k in range(chain.shape[-1]):
        tau = acor(np.mean(chain[:,:,k], axis=1))[0]
        taumax = max(tau,taumax)

    tau = int(np.ceil(taumax))

    return chain[::tau, :,:]

def tukey_window(N, alpha = 1.0/8.0):
    """Returns a normalized Tukey window function that tapers the first
    ``alpha*N/2`` and last ``alpha*N/2`` samples from a stretch of
    data.

    """

    istart = int(round(alpha*(N-1)/2.0 + 1))
    iend = int(round((N-1)*(1-alpha/2.0)))

    w = np.ones(N)
    ns = np.arange(0, N)

    w[:istart] = 0.5*(1.0 + np.cos(np.pi*(2.0*ns[:istart]/(alpha*(N-1))-1)))
    w[iend:] = 0.5*(1.0 + np.cos(np.pi*(2.0*ns[iend:]/(alpha*(N-1)) - 2.0/alpha + 1.0)))

    wnorm = np.sqrt(np.sum(np.square(w))/N)
    
    return w/wnorm
                      
def norm_logpdf(xs, loc=0, scale=1):
    """Returns the log of the normal distribution's PDF at the given
    values.

    """

    args = (xs - loc)/scale

    return -0.91893853320467274178 - np.log(scale) - 0.5*args*args
