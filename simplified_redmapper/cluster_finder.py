from astropy.table import Table
import numpy as np
from .nfw_solver import lambda_solver

class ClusterFinder:
    '''
    Identify clusters in mock data.
    '''
    def __init__(self, galaxy, sort_key, d_proj, central_idx=None, \
                periodic=True, boxsize=0.0, max_r=3.0, background=None):
        '''
        Parameters
        ------------
        galaxy
            Input galaxy catalog. The first three columns should be 
            coordinates ['x', 'y', 'z'] and the catalog will be projected 
            along z-axis.
        sort_key : str
            Galaxies will be sorted according to sort_key of the catalog 
            for the initial percolation.
        d_proj : float
            Projection length. Cluster finder will see any galaxies within 
            d_c +- d_proj as members. It should use the same units as 
            coordinates.
        central_idx
            Index of central galaxies to be considered as potential 
            centers. If set to None, will consider all input galaxies.
        periodic : bool
            Whether the catalog has periodic boundary condition.
        boxsize : float
            Boxsize of the mock data. Used when dealing with periodic 
            boundary conditions.
        max_r : float
            Only galaxies within max_r of central galaxy will be considered
            in indentification.
        background : float
            Background galaxy density. If None, will assume V=boxsize**3 to 
            calculate.
        '''
        self.galaxy = Table(galaxy).copy()
        self.ng = len(self.galaxy)
        self.dproj = d_proj
        self.periodic=periodic
        self.boxsize=boxsize
        self.max_r = max_r
        self.sort_key = sort_key

        self.galaxy.add_columns([np.ones(self.ng, dtype=float), np.arange(self.ng)], \
                                names=['pfree', 'ID'])

        if central_idx is None:
            self.init_central = self.galaxy.copy()
        else:
            self.init_central = self.galaxy[central_idx].copy()
        self.init_central.remove_columns(['pfree'])
        self.init_central.add_column(np.zeros(len(self.init_central)), name='lambda')
        self.init_central.sort(sort_key, reverse=True)
        
        self.galaxy.add_index('ID')
        if background is None:
            self.background = self.ng/self.boxsize**3 * 2*self.dproj
        else:
            self.background = background
        
    def initial_run(self, init_aperture=0.5):
        assigned = set()
        N_init_c = len(self.init_central)
        for i in range(N_init_c):
            print('Initial run... Finished:{}; Total:{}'.format(i, N_init_c), end='\r')
            if self.init_central[i]['ID'] in assigned:
                continue
            temp_cluster = self.init_central[i]
            temp_galaxy = self.galaxy[self.galaxy['pfree'] != 0]
            temp_member = cylinder_filter(temp_galaxy, temp_cluster['pos'], \
                init_aperture, self.dproj, periodic=self.periodic, boxsize=self.boxsize)
            self.init_central['lambda'][i] = len(temp_member)
            assigned = assigned | set(temp_member['ID'])
        self.init_central = self.init_central[self.init_central['lambda'] >= 3]
        self.init_central.sort('lambda', reverse=True)
        self.init_central.add_index('ID')
    
    def percolation(self, compare_key=None):
        central_to_process = self.init_central.copy()
        # Identify the first cluster
        _central = central_to_process[0]
        f_cat, f_central = self.identify_single(_central, compare_key)
        f_central['cID'], f_cat['cID'] = 0, 0
        self.final_central = [f_central]
        self.final_member = [f_cat]
        tmem_gal = self.galaxy.loc[f_cat['ID']]
        tmem_gal['pfree'] = tmem_gal['pfree'] * (1 - f_cat['pmem'])
        central_to_process.remove_row(0)
        # Remove the potential centers having low pfree.
        low_pfree_id = tmem_gal[tmem_gal['pfree'] < 0.5]['ID']
        low_pfree_cluster = np.intersect1d(low_pfree_id, central_to_process['ID'])
        if len(low_pfree_cluster) > 0:
            low_pfree_cidx = central_to_process.loc_indices[low_pfree_cluster]
            central_to_process.remove_rows(low_pfree_cidx)

        # Loop over the list and do percolation.
        while len(central_to_process) > 0:
            N_central = len(self.final_central)
            print('Percolation... Finished:{}; Remaining:{}'.format(N_central, len(central_to_process)), end='\r')
            _central = central_to_process[0]
            f_cat, f_central = self.identify_single(_central, compare_key)
            f_central['cID'], f_cat['cID'] = N_central, N_central
            self.final_central.append(f_central)
            self.final_member.append(f_cat)
            tmem_gal = self.galaxy.loc[f_cat['ID']]
            tmem_gal['pfree'] = tmem_gal['pfree'] * (1 - f_cat['pmem'])
            central_to_process.remove_row(0)

            low_pfree_id = tmem_gal[tmem_gal['pfree'] < 0.5]['ID']
            low_pfree_cluster = np.intersect1d(low_pfree_id, central_to_process['ID'])
            if len(low_pfree_cluster) > 0:
                low_pfree_cidx = central_to_process.loc_indices[low_pfree_cluster]
                central_to_process.remove_rows(low_pfree_cidx)

    def identify_single(self, init_central, compare_key=None):
        if compare_key is None:
            compare_key = self.sort_key
        # Only consider galaxies close enough to the central.
        temp_central = Table(init_central)
        temp_cat = cylinder_filter(self.galaxy, temp_central['pos'], self.max_r, self.dproj)
        temp_cat.add_column(0.0, name='pmem')
        solver = lambda_solver(temp_cat['R'], temp_cat['pfree'], self.background)
        # Galaxy must be outside of Rc if its pmem=0 and vice versa.
        temp_cat = temp_cat[solver['pmem'] > 0]
        # Iteration until the central galaxy is the most dominant one within Rc.
        while temp_cat[compare_key].max() != temp_central[compare_key]:
            temp_central = Table(temp_cat[temp_cat[compare_key].argmax()])
            temp_central.remove_columns(['pfree', 'R'])
            temp_central.add_column(0.0, name='lambda')
            temp_cat = cylinder_filter(self.galaxy, temp_central['pos'], self.max_r, self.dproj)
            temp_cat.add_column(0.0, name='pmem')
            solver = lambda_solver(temp_cat['R'], temp_cat['pfree'], self.background)
            temp_cat = temp_cat[solver['pmem'] > 0]

        temp_cat['pmem'] = solver['pmem'][solver['pmem'] > 0]
        temp_central['lambda'] = solver['lambda']
        temp_central.add_column(-1, name='cID')
        temp_cat.add_column(-1, name='cID')
        return temp_cat, temp_central

def cylinder_filter(catalog, center, R0, d0, periodic=True, boxsize=0):
    '''
    Return the galaxies within the cylinder of (radius, height)=(R0, 2*d0).
    '''
    pos = catalog['pos'] - center
    # Correction for boundary conditions.
    if periodic and (boxsize > 0) and \
        (np.abs(center - boxsize/2).max() > boxsize/2 - R0):
        for axis in range(3):
            idx1, = np.nonzero(pos[:, axis] < -boxsize/2)
            pos[:, axis][idx1] += boxsize
            idx2, = np.nonzero(pos[:, axis] > boxsize/2)
            pos[:, axis][idx2] -= boxsize
    
    # R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    R = np.linalg.norm(pos[:, :2], axis=1)
    idx, = np.nonzero((R < R0) & (np.abs(pos[:, 2]) < d0))
    # idx, = np.nonzero(np.all(np.abs(pos[:, :2]) < R0, axis=1) & (np.abs(pos[:, 2]) < d0))
    result = catalog[idx].copy()
    # R = np.linalg.norm(pos[idx, :2], axis=1)
    result.add_column(R[idx], name='R')
    return result