import numpy as np
from scipy.optimize import root_scalar
from numba import jit, float64

R0 = 1 # in h^-1 Mpc physical
beta = 0.2
Rs = 0.15 # h^-1 Mpc physical

@jit(float64[:](float64[:], float64), nopython=True)
def p_NFW(R, Rc):
    '''
    Projected NFW profile from Bartelmann 1996.
    Truncated at Rc.

    Parameters
    -------------
    R : float
        Projected distance to the center of the halo.
    Rc : float
        Aperture adopted by the halo finder.
    '''
    # R = np.array(R).reshape(-1)
    x = R/Rs
    x[R < 0.1] = 0.1/Rs # To avoid singularity at R=0, set Sigma(R<0.1) = Sigma(0.1).
    Sigma = np.zeros(x.size)
    condition = (x < 1)
    x_in = x[condition]
    x_out= x[~condition]

    Sigma[condition] = 1 / (x_in**2 - 1) * \
                (1 - 2/np.sqrt(1 - x_in**2) * np.arctanh(np.sqrt((1 - x_in)/(1 + x_in))))
    Sigma[~condition] = 1 / (x_out**2 - 1) * \
                (1 - 2/np.sqrt(x_out**2 - 1) * np.arctan(np.sqrt((x_out - 1)/(1 + x_out))))
    Sigma[R > Rc] = 0 # Truncated at Rc.

    # Calculate normalization factor according to Eq.10 in Rykoff et al. 2012.
    rho = np.log(Rc)
    k_nfw = np.exp(1.6517 - 0.5479*rho + 0.1382*rho**2 - 0.0719*rho**3 - \
            0.01582*rho**4 - 0.00085499*rho**5)
    Sigma *= k_nfw
    return Sigma

@jit(float64(float64), nopython=True)
def R_c(rich):
    return R0 * (rich/100)**beta

@jit(float64[:](float64, float64[:], float64), nopython=True)
def pmem(rich, R, b):
    '''
    Calculate the membership propablities of input galaxies.
    Fomula: pmem = lmabda*u/(lambda*u + b)
    u = 2*pi*R * Sigma; b = 2*pi*R * b0

    Parameters
    -----------
    rich : float
        Richness of the cluster.
    R : numpy.array
        Distances of input galaxies.
    b : float
        Background density.
    '''
    # R = np.array(R).reshape(-1)
    # print(R)
    Rc = R_c(rich)
    # u = 2*np.pi * R * p_NFW(R, Rc)
    return rich*p_NFW(R, Rc) / (rich*p_NFW(R, Rc) + b)

@jit(float64(float64[:], float64[:]), nopython=True)
def lambda_calculator(pfree, pmem):
    return np.sum(pfree * pmem)

@jit(float64(float64, float64[:], float64, float64[:]), nopython=True)
def func(rich, R, b, pfree):
    '''
    Function to solve for richness.
    Equation: SUM(pfree*u/(rich*u + b)) - 1
    '''
    return lambda_calculator(pfree, pmem(rich, R, b)) - rich

def lambda_solver(R, pfree, b, limit=[0.1, 1e3], return_pmem=True):
    '''
    Solve for richness of a cluster.
    Equation: 
    '''
    result = {}
    sol = root_scalar(func, args=(R, b, pfree), method='bisect', bracket=limit, \
        x0=20, rtol=1e-3)
    result['lambda'] = sol.root
    if return_pmem:
        p_mem = pmem(result['lambda'], R, b)
        result['pmem'] = p_mem
    return result