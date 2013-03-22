import numpy as np

def mc_eta_to_masses(mc, eta):
    mtot = mc/eta**0.6
    mprod = eta*mtot*mtot

    disc = mtot*mtot - 4.0*mprod

    sqrt_disc = np.sqrt(disc)

    m1 = 0.5*(mtot + sqrt_disc)
    m2 = 0.5*(mtot - sqrt_disc)

    return (m1,m2)

def masses_to_mc_eta(m1, m2):
    mtot = m1+m2
    eta = m1*m2/(mtot*mtot)
    mc = mtot*eta**0.6

    return (mc,eta)
