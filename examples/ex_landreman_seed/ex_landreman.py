#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
from simsopt.field.biotsavart import BiotSavart
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius
from mpi4py import MPI
import argparse
comm = MPI.COMM_WORLD
rank = comm.rank


def printf(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument("--length")
parser.add_argument("--seed")
args = parser.parse_args()

if args.length:
    length=float(args.length)
if args.seed:
    seed=int(args.seed)



#boozer_surface_list, base_curves, base_currents, coils = pys.compute_surfaces_in_landreman(mpol=10, ntor=10, exact=False, Nt_coils=16, write_to_file=True, length=18)
boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_landreman(mpol=10, ntor=10, exact=False, idx_surfaces=[rank], Nt_coils=16, length=18, time_stamp='1651158336.1647642', verbose=True)


# you can either fix the current in a single coil or introduce a toroidal flux constraint to prevent
# currents from going to zero.  We do the former here:
def fix_all_dofs(optims):
    if not isinstance(optims, list):
        optims = [optims]
    for o in optims:
        for a in o._get_ancestors():
            a.fix_all()
fix_all_dofs(base_currents[0])

if seed > 0:
    np.random.seed(seed)
    order = 4
    idx = np.arange(0,2*order+1)
    for bc in base_curves:
        coil_dofs = bc.x.copy()
        coil_dofs = coil_dofs.reshape((3, 33))
        coil_dofs[:, idx] += 2*(np.random.rand(3,idx.size)-0.5) * 0.05
        coil_dofs = coil_dofs.flatten()
        bc.x = coil_dofs




############################################################################
## SET THE TARGET IOTAS, MAJOR RADIUS, TOROIDAL FLUX                      ##
############################################################################

iotas_target = [None for bs in boozer_surface_list]
mr_target   = [None for bs in boozer_surface_list]
J_inner_radius = MajorRadius(boozer_surface_list[0])

if rank == 0:
    mr_target[0] = J_inner_radius.J()
    iotas_target[0] = boozer_surface_list[0].res['iota']

if rank == 3:
    iotas_target[0] = boozer_surface_list[0].res['iota']

KAPPA_MAX = 5.
MSC_MAX = 5.
LENGTHBOUND = 18
MIN_DIST = 0.10
IOTAS_AVG_TARGET = -0.42
ALEN_THRESHOLD = 1e-3

############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################

KAPPA_WEIGHT = 1e-9
MSC_WEIGHT = 1e-9
LENGTHBOUND_WEIGHT = 1e-9
MIN_DIST_WEIGHT = 1e-3
ALEN_WEIGHT = 1e-4
IOTAS_TARGET_WEIGHT = 1.
IOTAS_AVG_WEIGHT = 1.
MR_WEIGHT = 1.
RESIDUAL_WEIGHT = 1.

problem = pys.SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                             iotas_target=iotas_target, iotas_avg_target=IOTAS_AVG_TARGET, major_radii_targets=mr_target, 
                             iotas_target_weight=IOTAS_TARGET_WEIGHT, iotas_avg_weight=IOTAS_AVG_WEIGHT, mr_weight=MR_WEIGHT,
                             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                             msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                             residual_weight=RESIDUAL_WEIGHT,
                             rank=rank, outdir_append=f"seed={seed}_len={length}/init")


dofs = problem.x.copy()
dofs_prev=dofs.copy()
problem.callback(dofs)

def J_scipy(dofs,*args):
    problem.x = dofs
    J = problem.J()
    dJ = problem.dJ()
    return J, dJ

from scipy.optimize import minimize

MAXITER = 1000
for outer_iter in range(15):
    dofs = dofs_prev.copy()
    
    printf(f"""
    ################################################################################
    Outer iteration {outer_iter}
    ################################################################################
    """)

    problem = pys.SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                                 iotas_target=iotas_target, iotas_avg_target=IOTAS_AVG_TARGET, major_radii_targets=mr_target, 
                                 iotas_target_weight=IOTAS_TARGET_WEIGHT, iotas_avg_weight=IOTAS_AVG_WEIGHT, mr_weight=MR_WEIGHT,
                                 minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                                 msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                                 distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                                 residual_weight=RESIDUAL_WEIGHT,
                                 rank=rank, outdir_append=f"seed={seed}_len={length}/outer_iter={outer_iter}")

    res = minimize(J_scipy, dofs, jac=True, method='bfgs', tol=1e-20, callback=problem.callback, options={'maxiter': MAXITER})
    dofs_prev = res.x.copy()

    if (np.sum([J.J() for J in problem.J_coil_lengths])-LENGTHBOUND)/LENGTHBOUND > 0.001:
        LENGTHBOUND_WEIGHT*=10
    if (MIN_DIST-problem.J_distance.shortest_distance())/MIN_DIST > 0.001:
        MIN_DIST_WEIGHT*=10
    if (np.max([c.kappa() for c in base_curves]) - KAPPA_MAX)/KAPPA_MAX > 0.001:
        KAPPA_WEIGHT*=10
    if (np.max([J.J() for J in problem.J_msc])-MSC_MAX)/MSC_MAX > 0.001:
        MSC_WEIGHT*=10
    if sum(J.J() for J in problem.J_arclength) > ALEN_THRESHOLD:
        ALEN_WEIGHT*=10
    np.savetxt(problem.outdir + f"x_{outer_iter}.txt", res.x)
    np.savetxt(problem.outdir + f"weights_{outer_iter}.txt", [LENGTHBOUND_WEIGHT, MIN_DIST_WEIGHT, KAPPA_WEIGHT, MSC_WEIGHT, ALEN_WEIGHT])
    printf(f"J{outer_iter}={res.fun}, ||dJ||={np.linalg.norm(res.jac, ord=np.inf)}\n")
    printf(f"NEW WEIGHTS = [{LENGTHBOUND_WEIGHT}, {MIN_DIST_WEIGHT}, {KAPPA_WEIGHT}, {MSC_WEIGHT}, {ALEN_WEIGHT}]")
    printf(f"{res['success']}, {res['message']}")
