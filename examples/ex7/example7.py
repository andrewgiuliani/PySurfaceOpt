#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
import os
from lbfgs import fmin_lbfgs
from simsopt.geo.surfaceobjectives import ToroidalFlux, MajorRadius
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength
from pysurfaceopt.logging import info
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(Nt_coils=12, idx_surfaces=[0, 1, 2, 3], exact=False, time_stamp='1632415872.6999981')
boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX_half(Nt_coils=12, idx_surfaces=[0, 1, 2, 3], exact=False, time_stamp='1636473777.0197566')

bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
J_inner_toroidal_flux = ToroidalFlux(boozer_surface_list[0].surface, bs_tf, stellarator, boozer_surface=boozer_surface_list[0]) 
J_inner_radius = MajorRadius(boozer_surface_list[0], stellarator)

iotas_target = [None for bs in boozer_surface_list]
iotas_target[0] = boozer_surface_list[0].res['iota']
iotas_target[3] = boozer_surface_list[3].res['iota']
#iotas_target = [bs.res['iota'] for bs in boozer_surface_list]
mr_target   = [None for bs in boozer_surface_list]
tf_target   = [None for bs in boozer_surface_list]
mr_target[0] = J_inner_radius.J()
tf_target[0] = J_inner_toroidal_flux.J()
initial_areas = [boozer_surface.surface.area() for boozer_surface in boozer_surface_list]

############################################################################
## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
############################################################################

KAPPA_MAX = 5.
KAPPA_WEIGHT = 1e-9

LENGTHBOUND = sum([CurveLength(c).J() for c in stellarator._base_coils])
LENGTHBOUND_WEIGHT = 1e-9

MIN_DIST = 0.15
MIN_DIST_WEIGHT = 1e-3

ALEN_WEIGHT = 1e-7

IOTAS_WEIGHT = 1.
MAJOR_RADIUS_WEIGHT = 1.
TF_WEIGHT = 1.
MR_WEIGHT = 1.

############################################################################
# SET UP INITIAL OPTIMIZATION OBJECT
############################################################################
if args.seed > 0:
    
    order = 4
    idx = np.arange(0,2*order+1)

    np.random.seed(int(args.seed))
    idx_curr = np.arange(0,len(stellarator.get_currents()))
    idx_coil = np.arange(idx_curr[-1]+1, idx_curr[-1]+1 + len(stellarator.get_dofs()))
    coil_dofs = stellarator.get_dofs()
    coil_dofs = coil_dofs.reshape((3,3,25))
    coil_dofs[:, :, idx] += 2*(np.random.rand(3,3,idx.size)-0.5) * 0.05
    coil_dofs = coil_dofs.flatten()
    stellarator.set_dofs(coil_dofs)


problem = pys.SurfaceProblem(boozer_surface_list, stellarator,
         initial_areas=initial_areas,
         iotas_target=iotas_target, major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
         iotas_weight=IOTAS_WEIGHT, mr_weight=MAJOR_RADIUS_WEIGHT, tf_weight=TF_WEIGHT,
         minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
         distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
         area_weight=0., outdir_append="_seed=" + str(args.seed)  + "_runID=-1")

coil_dofs_reference = problem.x.copy()
boozer_surface_dofs_reference = [{"dofs": boozer_surface.surface.get_dofs(),
                                  "iota": boozer_surface.res["iota"],
                                  "G": boozer_surface.res["G"]} for boozer_surface in problem.boozer_surface_list]



############################################################################
# SAVE REFERENCE COIL DOFS AND BOOZER_SURFACE DOFS
############################################################################

info("")
info("")
info("Starting the penalty cycling")
info(f"MIN_DIST={MIN_DIST:.6e} KAPPA_MAX={KAPPA_MAX:.6e} LENGTHBOUND_THRESHOLD={LENGTHBOUND:.6e}, IOTAS_TARGET={iotas_target}, MR_TARGET={mr_target}, TF_TARGET={tf_target}")
info("")
info("")

curiter = 0
outeriter = 0
PENINCREASES = 10
MAXLOCALITER = 3000
#MAXLOCALITER =1

while outeriter < 10:
    if outeriter > 0 and outeriter < PENINCREASES:
        mk = max([np.max(c.kappa()) for c in stellarator._base_coils])
        tot_len = sum([J.J() for J in problem.J_coil_lengths])
        md = problem.min_dist() 
        al = max([J7.J() for J7 in problem.J_arclength])
        iotas = max([np.abs((J2.J()-iotas_target)/iotas_target) if iotas_target is not None else 0. for J2, iotas_target in zip(problem.J_iotas, problem.iotas_target)])
        mr =  max([np.abs((J3.J() - l)/l) if l is not None else 0. for (J3, l) in zip(problem.J_major_radii, problem.major_radii_targets)])
        tf = max([np.abs((J6.J() - l)/l) if l is not None else 0. for (J6, l) in zip(problem.J_toroidal_flux, problem.toroidal_flux_targets)]) 

        if mk > (1+1e-3)*KAPPA_MAX:
            KAPPA_WEIGHT *= 10.
            info(f"Increase weight for kappa {mk:.6e}")
        if tot_len > (1+1e-3)*LENGTHBOUND:
            LENGTHBOUND_WEIGHT *= 10.
            info(f"Increase weight for total coil length {tot_len:.6e}")
        if md < (1-1e-3)*MIN_DIST:
            MIN_DIST_WEIGHT *= 10.
            info(f"Increase weight for distance {md:.6e}")
        if al > 1e-3:
            ALEN_WEIGHT *= 10.
            info(f"Increase weight for arclen {al:.6e}")
        if iotas > 1e-3:
            IOTAS_WEIGHT*=10
            info(f"Increase weight for iotas {iotas:.6e}")
        if mr > 1e-3:
            MR_WEIGHT*=10
            info(f"Increase weight for major radius {mr:.6e}")
        if tf > 1e-3:
            TF_WEIGHT*=10
            info(f"Increase weight for toroidal flux {tf:.6e}")

        info(f"MIN_DIST_WEIGHT={MIN_DIST_WEIGHT:.6e} KAPPA_WEIGHT={KAPPA_WEIGHT:.6e} LENGTHBOUND_WEIGHT={LENGTHBOUND_WEIGHT:.6e}, ALEN_WEIGHT={ALEN_WEIGHT:.6e}, IOTAS_WEIGHT={IOTAS_WEIGHT:.6e}, MR_WEIGHT={MR_WEIGHT:.6e}, TF_WEIGHT={TF_WEIGHT:.6e}")
    
    problem = pys.SurfaceProblem(boozer_surface_list, stellarator,
             initial_areas=initial_areas,
             iotas_target=iotas_target, major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
             iotas_weight=IOTAS_WEIGHT, mr_weight=MAJOR_RADIUS_WEIGHT, tf_weight=TF_WEIGHT,
             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
             area_weight=0., outdir_append="_seed=" + str(args.seed) + f"_runID={outeriter}")
    
    for idx, booz in enumerate(problem.boozer_surface_list):
        problem.boozer_surface_reference[idx]['dofs'] = boozer_surface_dofs_reference[idx]['dofs']
        problem.boozer_surface_reference[idx]['iota'] = boozer_surface_dofs_reference[idx]['iota']
        problem.boozer_surface_reference[idx]['G'] = boozer_surface_dofs_reference[idx]['G']
    problem.update(coil_dofs_reference)
    
    def fun_pylbfgs(dofs,g,*args):
        problem.update(dofs)
        
        J = problem.res
        dJ = problem.dres
    
        g[:] = dJ
        return J
    
    def J_scipy(dofs,*args):
        problem.update(dofs)
        J = problem.res
        dJ = problem.dres
        return J, dJ
    
    
    info("NEW OUTER ITERATION STARTING")
    coeffs = problem.x.copy()
    maxiter = MAXLOCALITER
    from scipy.optimize import minimize
    res = minimize(J_scipy, coeffs, jac=True, method='bfgs', tol=1e-20, options={"maxiter": maxiter}, callback=problem.callback)
    info(f"{res['success']}, {res['message']}")

    # double check that the coefficients output by scipy are the ones stored at the previous callback (or at initialization of the object) 
    # assert np.allclose(res['x'], problem.x_reference, rtol=0, atol=0)
    # revert to the previously accepted coil dofs in x_reference
    coil_dofs_reference = problem.x_reference.copy()
    # deep copy of the surface dofs to initialize the next outer loop problem
    boozer_surface_dofs_reference = [{"dofs": problem.boozer_surface_reference[idx]["dofs"].copy(),
                                      "iota": problem.boozer_surface_reference[idx]["iota"],
                                      "G":    problem.boozer_surface_reference[idx]["G"]} for idx in range(len(problem.boozer_surface_reference))]

    # the surfaces should automatically be set to their reference states
    problem.update(coil_dofs_reference)

    # save final coils, currents, and surfaces to file
    np.savetxt(problem.outdir + 'final_BoozerLS_x.txt', problem.x)
    for idx, booz in enumerate(problem.boozer_surface_list):
        np.savetxt(problem.outdir + f'final_surface_BoozerLS_{idx}.txt', np.concatenate( (problem.boozer_surface_reference[idx]['dofs'], [problem.boozer_surface_reference[idx]['iota'], problem.boozer_surface_reference[idx]['G']]) ) )
    info("Final BoozerLS callback:")
    problem.callback(problem.x)
    problem.plot('final_BoozerLS.png')





    maxnorm = np.linalg.norm(res['jac'], ord=np.inf)
    if res['message'] == 'Desired error not necessarily achieved due to precision loss.' and np.linalg.norm(res['jac'], ord=np.inf) > 1e-7:
        info(f'Line search failed! Desired error not necessarily achieved due to precision loss. and max norm of gradient is {maxnorm}')
        break
    
    elif res['message'] == 'Desired error not necessarily achieved due to precision loss.' and np.linalg.norm(res['jac'], ord=np.inf) <= 1e-7:
        info(f'Line search failed! Desired error not necessarily achieved due to precision loss. and max norm of gradient is {maxnorm}')
    
    elif res['message'] == 'Maximum number of iterations has been exceeded.':
        info(f'Maximum number of iterations has been exceeded. and max norm of gradient is {maxnorm}')
    
    else:
        info('Unrecognized message')

    curiter += res.nit
    outeriter+=1




info("CHANGING TO BOOZEREXACT")
## CHANGE TO BoozerExact
#boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX(idx_surfaces=[0, 1, 2, 3], Nt_coils=12, exact=True, time_stamp='1632417037.222326')
boozer_surface_list, coils, currents, ma, stellarator = pys.load_surfaces_in_NCSX_half(Nt_coils=12, idx_surfaces=[0, 1, 2, 3], exact=True, time_stamp='1636472072.192064')


curiter = 0
outeriter = 0
PENINCREASES = 10
MAXLOCALITER = 3000
#MAXLOCALITER = 1

while outeriter < 15:
    if outeriter > 0 and outeriter < PENINCREASES:
        mk = max([np.max(c.kappa()) for c in stellarator._base_coils])
        tot_len = sum([J.J() for J in problem.J_coil_lengths])
        md = problem.min_dist() 
        al = max([J7.J() for J7 in problem.J_arclength])
        iotas = max([np.abs((J2.J()-iotas_target)/iotas_target) if iotas_target is not None else 0. for J2, iotas_target in zip(problem.J_iotas, problem.iotas_target)])
        mr =  max([np.abs((J3.J() - l)/l) if l is not None else 0. for (J3, l) in zip(problem.J_major_radii, problem.major_radii_targets)])
        tf = max([np.abs((J6.J() - l)/l) if l is not None else 0. for (J6, l) in zip(problem.J_toroidal_flux, problem.toroidal_flux_targets)]) 

        if mk > (1+1e-3)*KAPPA_MAX:
            KAPPA_WEIGHT *= 10.
            info(f"Increase weight for kappa {mk:.6e}")
        if tot_len > (1+1e-3)*LENGTHBOUND:
            LENGTHBOUND_WEIGHT *= 10.
            info(f"Increase weight for total coil length {tot_len:.6e}")
        if md < (1-1e-3)*MIN_DIST:
            MIN_DIST_WEIGHT *= 10.
            info(f"Increase weight for distance {md:.6e}")
        if al > 1e-3:
            ALEN_WEIGHT *= 10.
            info(f"Increase weight for arclen {al:.6e}")
        if iotas > 1e-3:
            IOTAS_WEIGHT*=10
            info(f"Increase weight for iotas {iotas:.6e}")
        if mr > 1e-3:
            MR_WEIGHT*=10
            info(f"Increase weight for major radius {mr:.6e}")
        if tf > 1e-3:
            TF_WEIGHT*=10
            info(f"Increase weight for toroidal flux {tf:.6e}")

        info(f"MIN_DIST_WEIGHT={MIN_DIST_WEIGHT:.6e} KAPPA_WEIGHT={KAPPA_WEIGHT:.6e} LENGTHBOUND_WEIGHT={LENGTHBOUND_WEIGHT:.6e}, ALEN_WEIGHT={ALEN_WEIGHT:.6e}, IOTAS_WEIGHT={IOTAS_WEIGHT:.6e}, MR_WEIGHT={MR_WEIGHT:.6e}, TF_WEIGHT={TF_WEIGHT:.6e}")
    
    problem = pys.SurfaceProblem(boozer_surface_list, stellarator,
             initial_areas=initial_areas,
             iotas_target=iotas_target, major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
             iotas_weight=IOTAS_WEIGHT, mr_weight=MAJOR_RADIUS_WEIGHT, tf_weight=TF_WEIGHT,
             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
             area_weight=0., outdir_append="_seed=" + str(args.seed) + f"_runID={outeriter}")
    
    for idx, booz in enumerate(problem.boozer_surface_list):
        problem.boozer_surface_reference[idx]['dofs'] = boozer_surface_dofs_reference[idx]['dofs']
        problem.boozer_surface_reference[idx]['iota'] = boozer_surface_dofs_reference[idx]['iota']
        problem.boozer_surface_reference[idx]['G'] = boozer_surface_dofs_reference[idx]['G']
    problem.update(coil_dofs_reference)

    def fun_pylbfgs(dofs,g,*args):
        problem.update(dofs)
        
        J = problem.res
        dJ = problem.dres
    
        g[:] = dJ
        return J
    
    def J_scipy(dofs,*args):
        problem.update(dofs)
        J = problem.res
        dJ = problem.dres
        return J, dJ
    
    
    info("NEW OUTER ITERATION STARTING")
    coeffs = problem.x.copy()
    maxiter = MAXLOCALITER
    from scipy.optimize import minimize
    res = minimize(J_scipy, coeffs, jac=True, method='bfgs', tol=1e-20, options={"maxiter": maxiter}, callback=problem.callback)
    info(f"{res['success']} {res['message']}")

    # double check that the coefficients output by scipy are the ones stored at the previous callback (or at initialization of the object) 
    #assert np.allclose(res['x'], problem.x_reference, rtol=0, atol=0)
    coil_dofs_reference = problem.x_reference.copy()
    # deep copy of the surface dofs to initialize the next outer loop problem
    boozer_surface_dofs_reference = [{"dofs": problem.boozer_surface_reference[idx]["dofs"].copy(),
                                      "iota": problem.boozer_surface_reference[idx]["iota"],
                                      "G":    problem.boozer_surface_reference[idx]["G"]} for idx in range(len(problem.boozer_surface_reference))]

    #surface dofs should always be the reference one
    problem.update(coil_dofs_reference)

    # save final coils, currents, and surfaces to file
    np.savetxt(problem.outdir + 'final_BoozerExact_x.txt', problem.x)
    for idx, booz in enumerate(problem.boozer_surface_list):
        np.savetxt(problem.outdir + f'final_surface_BoozerExact_{idx}.txt', np.concatenate( (problem.boozer_surface_reference[idx]['dofs'], [problem.boozer_surface_reference[idx]['iota'], problem.boozer_surface_reference[idx]['G']]) ) )
    info("Final BoozerExact callback:")
    problem.callback(problem.x)
    problem.plot('final_BoozerExact.png')

    maxnorm = np.linalg.norm(res['jac'], ord=np.inf)
    if res['message'] == 'Desired error not necessarily achieved due to precision loss.' and np.linalg.norm(res['jac'], ord=np.inf) > 1e-7:
        info(f'Line search failed! Desired error not necessarily achieved due to precision loss. and max norm of gradient is {maxnorm}')
        break
    
    elif res['message'] == 'Desired error not necessarily achieved due to precision loss.' and np.linalg.norm(res['jac'], ord=np.inf) <= 1e-7:
        info(f'Line search failed! Desired error not necessarily achieved due to precision loss. and max norm of gradient is {maxnorm}')
    
    elif res['message'] == 'Maximum number of iterations has been exceeded.':
        info(f'Maximum number of iterations has been exceeded. and max norm of gradient is {maxnorm}')
    
    else:
        info('Unrecognized message')


    curiter += res.nit
    outeriter+=1







coeffs = problem.x_reference.copy()
#coeffs = res.x.copy()
#assert np.allclose(coeffs, problem.x_reference, rtol=0, atol=0)


# save final coils, currents, and surfaces to file
np.savetxt(problem.outdir + 'before_newton_x.txt', problem.x)
for idx, booz in enumerate(problem.boozer_surface_list):
    np.savetxt(problem.outdir + f'before_newton_surface_{idx}.txt', np.concatenate( (problem.boozer_surface_reference[idx]['dofs'], [problem.boozer_surface_reference[idx]['iota'], problem.boozer_surface_reference[idx]['G']]) ) )






h = np.random.rand(coeffs.shape[0])
## EXECUTE A TAYLOR TEST TO VERIFY GRADIENT AT EXIT POINT IS CORRECT .... ##
def taylor_test(obj, x, order=6, h=None):
    if h is None:
        np.random.seed(1)
        h = np.random.rand(x.shape[0])
    obj.update(x, verbose=False)
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
        obj.update(x + shifts[0]*eps*h, verbose=False)
        fd = weights[0] * obj.res
        for i in range(1, len(shifts)):
            obj.update(x + shifts[i]*eps*h, verbose=False)
            fd += weights[i] * obj.res
        err = abs(fd/eps - djh)
        pys.info("%.6e, %.6e, %.6e", eps, err, err/np.linalg.norm(djh))
    obj.update(x, verbose=False)
    pys.info("-----")

taylor_test(problem, coeffs, order=6, h=h)
np.savetxt(problem.outdir + 'taylor_test_coeffs.txt', coeffs)
np.savetxt(problem.outdir + 'taylor_test_h.txt', h)











def fun(x, verbose=True):
    problem.update(x, verbose=verbose)
    return problem.res, problem.dres

def approx_H(x, eps=1e-4):
    n = x.size
    H = np.zeros((n, n))
    #return np.eye(n)
    x0 = x
    for i in range(n):
        x = x0.copy()
        x[i] += eps
        d1 = fun(x, verbose=False)[1]
        x[i] -= 2*eps
        d0 = fun(x, verbose=False)[1]
        H[i, :] = (d1-d0)/(2*eps)
    HH = 0.5 * (H+H.T)
    return H, HH



from scipy.linalg import eigh
x = problem.x.copy()
f, d = fun(x)
eps = 1e-4
for i in range(5):
    try:
        Hnonsym, H = approx_H(x, eps=eps)
        D, E = eigh(H)
    except:
        print(f"Newton iteration {i} failed, decrease eps")
        eps *= 0.1
        continue
    bestd = np.inf
    bestx = None
    # Computing the Hessian is the most expensive thing, so we can be pretty
    # naive with the next step and just try a whole bunch of damping parameters
    # and step sizes and then take the one with smallest gradient norm that
    # still decreases the objective
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        Dm = np.abs(D) + lam
        dx = E @ np.diag(1./Dm) @ E.T @ d
        alpha = 1.
        for j in range(5):
            xnew = x - alpha * dx
            fnew, dnew = fun(xnew, verbose=False)
            dnormnew = np.linalg.norm(dnew, ord=np.inf)
            foundnewbest = ""
            if fnew < f and dnormnew < bestd:
                bestd = dnormnew
                bestx = xnew
                foundnewbest = "x"
            print(f'Linesearch: lam={lam:.5f}, alpha={alpha:.4f}, J(xnew)={fnew:.15f}, |dJ(xnew)|={dnormnew:.3e}, {foundnewbest}')
            alpha *= 0.5
    if bestx is None:
        print(f"Stop Newton because no point with smaller function value could be found.")
        break
    fnew, dnew = fun(bestx)
    dnormnew = np.linalg.norm(dnew, ord=np.inf)
    if dnormnew >= np.linalg.norm(d, ord=np.inf):
        print(f"Stop Newton because |{dnormnew}| >= |{np.linalg.norm(d, ord=np.inf)}|.")
        break
    x = bestx
    d = dnew
    f = fnew
    print(f"J(x)={f:.15f}, |dJ(x)|={np.linalg.norm(d, ord=np.inf):.3e}")

problem.update(x) 
info("After Newton callback:")
problem.callback(x)
problem.plot('final.png')

# save final coils, currents, and surfaces to file
np.savetxt(problem.outdir + 'final_x.txt', problem.x)
for idx, booz in enumerate(problem.boozer_surface_list):
    np.savetxt(problem.outdir + f'final_surface_{idx}.txt', np.concatenate( (problem.boozer_surface_reference[idx]['dofs'], [problem.boozer_surface_reference[idx]['iota'], problem.boozer_surface_reference[idx]['G']]) ) )






## COMPUTE HESSIAN ##
print("Computing the Hessian now")
H, Hsym = approx_H(coeffs, eps=1e-4)

from scipy.linalg import eigh
from scipy.linalg import eig
w,v = eig(H)
wsym,vsym = eigh(Hsym)

np.savetxt(problem.outdir + 'w.txt', w)
np.savetxt(problem.outdir + 'wsym.txt', wsym)
