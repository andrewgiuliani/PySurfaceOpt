#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
import os

import pysurfaceopt as pys
from simsopt.geo.surfaceobjectives import ToroidalFlux, MajorRadius
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength
from pysurfaceopt.logging import info

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank


## more digits with fewer coil dofs I've noticed, try below test:
#vol_list = np.linspace(-0.162, -3.3, 10)
#boozer_surface_list, coils, currents, ma, stellarator = pys.compute_surfaces_in_NCSX(mpol=10, ntor=10, Nt_coils=6, exact=False, write_to_file=False, vol_list=vol_list)
#boozer_surface_list = [boozer_surface_list[0], boozer_surface_list[3]]

boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(idx_surfaces=[rank], Nt_coils=12, exact=True, time_stamp='1632417037.222326')
#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[rank], exact=False, time_stamp='1632415872.6999981')

#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(idx_surfaces=[0, 3, 8], Nt_coils=12, exact=True, time_stamp='1632417037.222326')
#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[0, 3, 8], exact=False, time_stamp='1632415872.6999981')




bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
J_inner_toroidal_flux = ToroidalFlux(boozer_surface_list[0].surface, bs_tf, stellarator, boozer_surface=boozer_surface_list[0]) 
J_inner_radius = MajorRadius(boozer_surface_list[0], stellarator)

iotas_target = [None for bs in boozer_surface_list]
mr_target   = [None for bs in boozer_surface_list]
tf_target   = [None for bs in boozer_surface_list]
mr_target[0] = J_inner_radius.J() + 0.01
tf_target[0] = J_inner_toroidal_flux.J() + 0.01
iotas_target[0] = boozer_surface_list[0].res['iota']+0.01
iotas_avg = -0.5


BOOZER_RESIDUAL_WEIGHT = [None for boozer_surface in boozer_surface_list]
BOOZER_RESIDUAL_WEIGHT[0] = 1.




############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################
MSC_MAX = 5.
MSC_WEIGHT = 1e-9


KAPPA_MAX = 5.
KAPPA_WEIGHT = 1e-9

LENGTHBOUND = sum([CurveLength(c).J() for c in stellarator._base_coils])-0.1
LENGTHBOUND_WEIGHT = 1e-2

MIN_DIST = 0.15
MIN_DIST_WEIGHT = 1e-3

ALEN_WEIGHT = 1e-3 

IOTAS_TARGET_WEIGHT = 1.
MAJOR_RADIUS_WEIGHT = 1. 
TF_WEIGHT = 1.
MR_WEIGHT = 1.
IOTAS_AVG_WEIGHT = 1.


initial_areas=[bs.surface.area() for bs in boozer_surface_list]


############################################################################
# SET UP INITIAL OPTIMIZATION OBJECT
############################################################################
problem = pys.SurfaceProblem(boozer_surface_list, stellarator,
#         initial_areas=initial_areas,
         iotas_target=iotas_target, major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
         iotas_target_weight=IOTAS_TARGET_WEIGHT, mr_weight=MAJOR_RADIUS_WEIGHT, tf_weight=TF_WEIGHT,
         iotas_avg_target=iotas_avg, iotas_avg_weight=IOTAS_AVG_WEIGHT,
         minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
         distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT, residual_weight=BOOZER_RESIDUAL_WEIGHT,
         msc_weight=MSC_WEIGHT, msc_max=MSC_MAX,
         outdir_append="taylor_test")

coeffs = problem.x.copy()
problem.callback(problem.x)


def taylor_test(obj, x, order=6, h=None, verbose=False):
    if h is None:
        np.random.seed(1)
        h = np.random.rand(x.shape[0])
    obj.update(x, verbose=verbose)
    dj0 = obj.dres
    djh = np.sum(dj0*h)
    if order == 1:
        shifts = [0, 1]
        weights = [-1, 1]
    elif order == 2:
        shifts = [-1, 1]
        weights = [-0.5, 0.5]
    elif order == 4:
        shifts = [-2, -1, 1, 2]
        weights = [1/12, -2/3, 2/3, -1/12]
    elif order == 6:
        shifts = [-3, -2, -1, 1, 2, 3]
        weights = [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]
    for i in range(10, 40):
        eps = 0.5**i
        obj.update(x + shifts[0]*eps*h, verbose=verbose)
        fd = weights[0] * obj.res
        for i in range(1, len(shifts)):
            obj.update(x + shifts[i]*eps*h, verbose=verbose)
            fd += weights[i] * obj.res
        err = abs(fd/eps - djh)
        pys.info("eps: %.6e, adjoint deriv: %.6e, fd deriv: %.6e, err: %.6e, rel. err:%.6e", eps, djh, fd/eps, err, err/np.linalg.norm(djh))
    obj.update(x, verbose=verbose)
    pys.info("-----")

taylor_test(problem, coeffs, order=6, verbose=False)
