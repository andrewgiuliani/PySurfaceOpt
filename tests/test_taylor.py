#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius, AreaPenalty, Volume
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank



#boozer_surface_list, base_curves, base_currents, coils = pys.compute_surfaces_in_NCSX(Nt_coils=12, exact=True, write_to_file=True)
#boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[0], exact=True, time_stamp='1636472072.192064')
#boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[rank], exact=True, time_stamp='1636472072.192064')
boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_landreman(mpol=10, ntor=10, exact=False, idx_surfaces=[rank], 
        Nt_coils=16, length=18, time_stamp='1651158336.1647642', verbose=True, weighting="1/B")



PEN_WEIGHT = 1e-4
for idx in range(len(boozer_surface_list)):
    sl = boozer_surface_list[idx].surface
    mpol = 5
    ntor = 5

    stemp = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=sl.stellsym, nfp=sl.nfp, quadpoints_phi=sl.quadpoints_phi, quadpoints_theta=sl.quadpoints_theta)
    stemp.least_squares_fit(sl.gamma())

    phis = np.linspace(0, 1/(2*sl.nfp), ntor+1+5, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1+5, endpoint=False)
    phis+=phis[0]/2
    shigh = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=sl.stellsym, nfp=sl.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    shigh.x = stemp.x
    s = shigh
    
    areatarget = s.area() 
    area = AreaPenalty(s, PEN_WEIGHT, areatarget*0.9) 
    penalty = area
    print(f"{rank}, area used={areatarget}")

    bs = BiotSavart(coils)
    ll = Volume(s)

    # need to actually store target surface label for BoozerLS surfaces
    target = boozer_surface_list[idx].targetlabel
    boozer_surface = BoozerSurface(bs, s, ll, target, reg=penalty)
    
    iota0 = boozer_surface_list[idx].res['iota']
    G0 = boozer_surface_list[idx].res['G']
    res = boozer_surface.minimize_boozer_penalty_constraints_BFGS(tol=1e-13, maxiter=10000, constraint_weight=100., iota=iota0, G=G0, weighting="1/B")
   
    boozer_surface.need_to_run_code = True
    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=1e-13, maxiter=30, constraint_weight=100., iota=res['iota'], G=res['G'], weighting="1/B")
    res['solver'] = 'NEWTON'
    print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}, cond={res['cond']:.3e}")
    boozer_surface_list[idx] = boozer_surface




#boozer_surface_list, base_curves, base_currents, coils = pys.compute_surfaces_in_landreman(mpol=10, ntor=10, exact=True, Nt_coils=16, write_to_file=False, vol_list=[-1.168750e-02], tol=1e-13, length=18)

#boozer_surface_list, base_curves, base_currents, coils = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[0], exact=False, time_stamp='1636473777.0197566')

############################################################################
## SET THE TARGET IOTAS, MAJOR RADIUS, TOROIDAL FLUX                      ##
############################################################################

iotas_target = [None for bs in boozer_surface_list]
mr_target   = [None for bs in boozer_surface_list]
J_inner_radius = MajorRadius(boozer_surface_list[0])

mr_target[0] = J_inner_radius.J()
iotas_target[0] = boozer_surface_list[0].res['iota']

############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################

KAPPA_MAX = 5.
KAPPA_WEIGHT = 1e-9

MSC_MAX = 5.
MSC_WEIGHT = 1e-9

LENGTHBOUND = 18.
LENGTHBOUND_WEIGHT = 1e-9

MIN_DIST = 0.15
MIN_DIST_WEIGHT = 1e-3

ALEN_WEIGHT = 1e-7

IOTAS_TARGET_WEIGHT = 1.
MR_WEIGHT = 1.
RES_WEIGHT = [1e4 for bs in boozer_surface_list]



problem = pys.SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                             #iotas_target=iotas_target,# major_radii_targets=mr_target, 
                             #iotas_target_weight=IOTAS_TARGET_WEIGHT,# mr_weight=MR_WEIGHT,
                             #minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                             #msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                             #distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                             residual_weight=RES_WEIGHT,
                             outdir_append="_runID=-1")
coeffs = problem.x.copy()
problem.callback(coeffs)

def taylor_test(obj, x, order=6, h=None, verbose=False):
    if h is None:
        np.random.seed(1)
        h = np.random.rand(x.shape[0])
    
    problem.x = x
    
    dj0 = obj.dJ()
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
    for i in range(12, 40):
        eps = 0.5**i
        obj.x = x + shifts[0]*eps*h
        fd = weights[0] * obj.J(verbose=True)
        for i in range(1, len(shifts)):
            obj.x = x + shifts[i]*eps*h
            fd += weights[i] * obj.J(verbose=True)
        err = abs(fd/eps - djh)
        print(f"eps: {eps:.6e}, adjoint deriv: {djh:.6e}, fd deriv: {fd/eps:.6e}, err: {err:.6e}, rel. err:{err/np.linalg.norm(djh):.6e}")
    obj.x = x
    print("-----")

taylor_test(problem, coeffs, order=6, verbose=False)
