#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
import os
from lbfgs import fmin_lbfgs
from simsopt.geo.surfaceobjectives import ToroidalFlux, MajorRadius
from simsopt.geo.biotsavart import BiotSavart


# This example loads BoozerExact and BoozerLS surfaces in the NCSX-like coil set for comparison

p0 = np.loadtxt('./initial_poincare_Nt_coils=12/poincare0.txt', delimiter=',')
p1 = np.loadtxt('./initial_poincare_Nt_coils=12/poincare1.txt', delimiter=',')
p2 = np.loadtxt('./initial_poincare_Nt_coils=12/poincare2.txt', delimiter=',')
p3 = np.loadtxt('./initial_poincare_Nt_coils=12/poincare3.txt', delimiter=',')

poincare = [p0, p1, p2, p3]




vol_list = np.linspace(-0.162, -3.3, 50)[:20]
#vol_list = [-1.3, -1.2, -1.1, -1.]
boozer_surface_list, coils, currents, ma, stellarator = pys.compute_surfaces_in_NCSX(mpol=11, ntor=11, Nt_coils=12, exact=True, write_to_file=False, vol_list=vol_list)

#import ipdb;ipdb.set_trace()
#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(mpol=10, ntor=10, idx_surfaces=np.arange(20), label="vol", Nt_coils=12, exact=True, time_stamp='1632168407.0710688')
tr = 300
cross = np.zeros((4, len(boozer_surface_list), tr, 3))
for idx, bs in enumerate(boozer_surface_list):
    s = bs.surface
    cross[0, idx, :, :] = s.cross_section(0.00*2*np.pi/3., thetas=tr)
    cross[1, idx, :, :] = s.cross_section(0.25*2*np.pi/3., thetas=tr)
    cross[2, idx, :, :] = s.cross_section(0.50*2*np.pi/3., thetas=tr)
    cross[3, idx, :, :] = s.cross_section(0.75*2*np.pi/3., thetas=tr)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for ii in range(4):
   
    row = ii // 2
    col = ii %  2
    axs[row, col].set_xlabel("$r$")
    axs[row, col].set_ylabel("$z$")
    axs[row, col].tick_params(direction="in")

    R = np.sqrt( poincare[ii][:,1]**2 + poincare[ii][:,2]**2)
    Z = poincare[ii][:,3]
    axs[row, col].scatter(R, Z, s=0.1)
 

    for idx in range(cross.shape[1]):
        R = np.sqrt(cross[ii,idx,:,0]**2 + cross[ii,idx,:,1]**2)
        Z = cross[ii,idx,:,2]
        R = np.concatenate((R,[R[0]]))
        Z = np.concatenate((Z,[Z[0]]))
        axs[row, col].plot(R,Z, linewidth = 1, color = (1,0,0))

plt.tight_layout()
plt.show()
plt.close()

#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(label="vol", Nt_coils=12, exact=False)
