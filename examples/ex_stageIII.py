#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
from simsopt.field.biotsavart import BiotSavart
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

length=18
if length == 18:
    boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_stageII(mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=[rank], exact=True, length=18, time_stamp='1639707710.6463501')
elif length == 20:
    boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_stageII(mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=[rank], exact=True, length=20, time_stamp='1639796640.5101252')
elif length == 22:
    boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_stageII(mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=[rank], exact=True, length=22, time_stamp='1642522947.7884622')
elif length == 24:
    boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_stageII(mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=[rank], exact=True, length=24, time_stamp='1642523475.5701194')
else:
    quit()

############################################################################
## SET THE TARGET IOTAS, MAJOR RADIUS, TOROIDAL FLUX                      ##
############################################################################

iotas_target = [None for bs in boozer_surface_list]
mr_target   = [None for bs in boozer_surface_list]
tf_target   = [None for bs in boozer_surface_list]

bs_tf = BiotSavart(coils)
J_inner_toroidal_flux = ToroidalFlux(boozer_surface_list[0], bs_tf) 
J_inner_radius = MajorRadius(boozer_surface_list[0])

if comm.size > 1:
    if rank == 0:
        mr_target[0] = J_inner_radius.J()
        tf_target[0] = J_inner_toroidal_flux.J()
else:
    mr_target[0] = J_inner_radius.J()
    tf_target[0] = J_inner_toroidal_flux.J()

############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################

KAPPA_MAX = 5.
KAPPA_WEIGHT = 1e-9

MSC_MAX = 5.
MSC_WEIGHT = 1e-9

LENGTHBOUND = length
LENGTHBOUND_WEIGHT = 1e-9

MIN_DIST = 0.1
MIN_DIST_WEIGHT = 1e-3

ALEN_WEIGHT = 1e-4

IOTAS_AVG_TARGET = -0.42
IOTAS_AVG_WEIGHT = 1.
TF_WEIGHT = 1.
MR_WEIGHT = 1.

problem = pys.SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                             major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
                             iotas_avg_weight=IOTAS_AVG_WEIGHT, iotas_avg_target=IOTAS_AVG_TARGET, mr_weight=MR_WEIGHT, tf_weight=TF_WEIGHT,
                             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                             msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                             rank=rank, outdir_append="")


coeffs = problem.x.copy()
problem.callback(coeffs)

def J_scipy(dofs,*args):
    problem.x = dofs
    J = problem.res
    dJ = problem.dres
    return J, dJ

from scipy.optimize import minimize
res = minimize(J_scipy, coeffs, jac=True, method='bfgs', tol=1e-20, callback=problem.callback)

