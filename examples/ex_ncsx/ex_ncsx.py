#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
from simsopt.field.biotsavart import BiotSavart
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank


boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[2*rank, 2*rank+1], exact=True, time_stamp='1636472072.192064', tol=1e-13)
#boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[0], exact=True, time_stamp='1636472072.192064', tol=1e-13)
#boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[0,1,2,3], exact=False, time_stamp='1636473777.0197566', tol=1e-10)
# you can either fix the current in a single coil or introduce a toroidal flux constraint to prevent
# currents from going to zero.  We do the former here:
base_currents[0].fix_all()

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
        iotas_target[0] = boozer_surface_list[0].res['iota']
    if rank == comm.size-1:
        iotas_target[-1] = boozer_surface_list[-1].res['iota']
else:
    mr_target[0] = J_inner_radius.J()
    tf_target[0] = J_inner_toroidal_flux.J()
    iotas_target[0] = boozer_surface_list[0].res['iota']
    iotas_target[-1] = boozer_surface_list[-1].res['iota']

############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################

KAPPA_MAX = 5.
KAPPA_WEIGHT = 1e-9

MSC_MAX = 5.
MSC_WEIGHT = 1e-9

LENGTHBOUND = 20.8
LENGTHBOUND_WEIGHT = 1e-9

MIN_DIST = 0.15
MIN_DIST_WEIGHT = 1e-3

ALEN_WEIGHT = 1e-4

IOTAS_TARGET_WEIGHT = 1.
TF_WEIGHT = 1.
MR_WEIGHT = 1.
RES_WEIGHT = 1e-3

problem = pys.SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                             iotas_target=iotas_target, major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
                             iotas_target_weight=IOTAS_TARGET_WEIGHT, mr_weight=MR_WEIGHT, tf_weight=TF_WEIGHT,
                             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                             msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                             residual_weight=RES_WEIGHT,
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

