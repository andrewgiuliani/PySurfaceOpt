#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
from simsopt.field.biotsavart import BiotSavart
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank


#boozer_surface_list, base_curves, base_currents, coils = pys.compute_surfaces_in_landreman(mpol=10, ntor=10, exact=False, Nt_coils=16, write_to_file=True, length=18)
boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_landreman(mpol=10, ntor=10, exact=False, idx_surfaces=[0], 
        Nt_coils=16, length=18, time_stamp='1651158336.1647642', verbose=True, weighting=None, hessian=False)
import ipdb;ipdb.set_trace()

# you can either fix the current in a single coil or introduce a toroidal flux constraint to prevent
# currents from going to zero.  We do the former here:
def fix_all_dofs(optims):
    if not isinstance(optims, list):
        optims = [optims]
    for o in optims:
        for a in o._get_ancestors():
            a.fix_all()
fix_all_dofs(base_currents[0])


############################################################################
## SET THE TARGET IOTAS, MAJOR RADIUS, TOROIDAL FLUX                      ##
############################################################################

iotas_target = [None for bs in boozer_surface_list]
mr_target   = [None for bs in boozer_surface_list]
J_inner_radius = MajorRadius(boozer_surface_list[0])

mr_target[0] = J_inner_radius.J()

if rank == 0 or rank == 3:
    iotas_target[0] = boozer_surface_list[0].res['iota']

############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################

KAPPA_MAX = 5.
KAPPA_WEIGHT = 1e-9

MSC_MAX = 5.
MSC_WEIGHT = 1e-9

LENGTHBOUND = 18
LENGTHBOUND_WEIGHT = 1e-9

MIN_DIST = 0.10
MIN_DIST_WEIGHT = 1e-3

ALEN_WEIGHT = 1e-4

IOTAS_TARGET_WEIGHT = 1.
MR_WEIGHT = 1.

problem = pys.SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                             iotas_target=iotas_target, major_radii_targets=mr_target, 
                             iotas_target_weight=IOTAS_TARGET_WEIGHT, mr_weight=MR_WEIGHT,
                             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                             msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                             rank=rank, outdir_append="BoozerLS")


coeffs = problem.x.copy()
problem.callback(coeffs)

def J_scipy(dofs,*args):
    problem.x = dofs
    J = problem.J()
    dJ = problem.dJ()
    return J, dJ

from scipy.optimize import minimize
res = minimize(J_scipy, coeffs, jac=True, method='bfgs', tol=1e-20, callback=problem.callback)
if rank == 0:
    print(f"{res['success']}, {res['message']}")
