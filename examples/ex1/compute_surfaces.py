#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
import os
from simsopt.geo.surfaceobjectives import ToroidalFlux, MajorRadius, Volume, boozer_surface_residual
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface

#####################################################################
## This example computes BoozerExact and in the NCSX-like coil set ##
#####################################################################


print("Computing BoozerExact surfaces")
vol_list = [-1.3, -1.2, -1.1, -1.]
boozer_surface_list1, coils, currents, ma, stellarator = pys.compute_surfaces_in_NCSX_half(mpol=11, ntor=11, Nt_coils=12, exact=True, write_to_file=False, vol_list=vol_list)
boozer_surface_list1 = [boozer_surface_list1[3]]

vol_list = np.linspace(-0.162, -3.3, 10)
vol_list = np.concatenate((vol_list[:3], vol_list[5:]))
boozer_surface_list2, coils, currents, ma, stellarator = pys.compute_surfaces_in_NCSX_half(mpol=11, ntor=11, Nt_coils=12, exact=True, write_to_file=False, vol_list=vol_list)
BoozerExact_list = boozer_surface_list1 + boozer_surface_list2

BoozerExact_labels = np.zeros((len(BoozerExact_list),))
for idx,surf in enumerate(BoozerExact_list):
    BoozerExact_labels[idx] = surf.label.J()

target1 = BoozerExact_labels[0]
dofs1 = np.concatenate((boozer_surface_list1[0].surface.get_dofs(), [boozer_surface_list1[0].res['iota'], boozer_surface_list1[0].res['G']]))
############################################################################################################
## This example computes BoozerLS and in the NCSX-like coil set using the BoozerExact surfaces at the ICs ##
############################################################################################################
print("Computing BoozerLS surfaces")
BoozerLS_list = [] 
for idx in range(len(BoozerExact_list)):
    stellsym=True
    nfp = 3
    ntor = 11
    mpol = 11
    nquadphi = ntor+1 + 5
    nquadtheta = 2*mpol+1 + 5
    phis = np.linspace(0, 1/(2*nfp), nquadphi, endpoint=False)
    thetas = np.linspace(0, 1, nquadtheta, endpoint=False)
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.set_dofs(BoozerExact_list[idx].surface.get_dofs())
    label = Volume(s, stellarator)
    target = BoozerExact_labels[idx]
    bs = BiotSavart(stellarator.coils, stellarator.currents)
    BoozerLS_list.append(BoozerSurface(bs, s, label, target))


    iota0 = BoozerExact_list[idx].res['iota']
    G0 = BoozerExact_list[idx].res['G']
    res = BoozerLS_list[idx].minimize_boozer_penalty_constraints_ls(tol=1e-9, maxiter=100, constraint_weight=1000., iota=iota0, G=G0, method='manual')
    bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
    tf = ToroidalFlux(s, bs_tf, stellarator)
    print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={label.J():.6e}, target = {target:.6e}, vol_rel_error={np.abs(label.J()-target)/np.abs(target):.6e}, ||residual||_2={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
target2 = BoozerExact_labels[0]
dofs2 = np.concatenate((BoozerLS_list[0].surface.get_dofs(), [BoozerLS_list[0].res['iota'], BoozerLS_list[0].res['G']]))

print("Tack on the additional surfaces")
# tack on the additional surfaces -1.3, -1.6
#for ll in [-1.3, -1.6]:
for ll in [-1.3, -1.5, -1.7]:
    stellsym=True
    nfp = 3
    ntor = 11
    mpol = 11
    nquadphi = ntor+1 + 5
    nquadtheta = 2*mpol+1 + 5
    phis = np.linspace(0, 1/(2*nfp), nquadphi, endpoint=False)
    thetas = np.linspace(0, 1, nquadtheta, endpoint=False)
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.set_dofs(BoozerExact_list[0].surface.get_dofs())
    label = Volume(s, stellarator)
    target = ll
    bs = BiotSavart(stellarator.coils, stellarator.currents)
    BoozerLS_list.append(BoozerSurface(bs, s, label, target))


    iota0 = BoozerExact_list[0].res['iota']
    G0 = BoozerExact_list[0].res['G']
    res = BoozerLS_list[-1].minimize_boozer_penalty_constraints_ls(tol=1e-9, maxiter=100, constraint_weight=1000., iota=iota0, G=G0, method='manual')
    bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
    tf = ToroidalFlux(s, bs_tf, stellarator)
    print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={label.J():.6e}, target = {target:.6e}, vol_rel_error={np.abs(label.J()-target)/np.abs(target):.6e}, ||residual||_2={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")


#####################################################################
## This example computes two BoozerExact surfaces with the same    ##
## label.                                                          ##
#####################################################################
print("Computing two BoozerExact solutions")
stellsym=True
nfp = 3
ntor = 11
mpol = 11
nquadphi = ntor+1
nquadtheta = 2*mpol+1
phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

s1 = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s1.set_dofs(dofs1[:-2])
iota0 = dofs1[-2]
G0 = dofs1[-1]
label1 = Volume(s1, stellarator)
bs1 = BiotSavart(stellarator.coils, stellarator.currents)
surf1 = BoozerSurface(bs, s1, label1, target1)
res = surf1.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=300,iota=iota0,G=G0)
r, = boozer_surface_residual(s1, res['iota'], res['G'], bs1, derivatives=0)
r_max = np.linalg.norm(r, ord=np.inf)
print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={surf1.label.J():.3f}, vol_rel_error={np.abs(surf1.label.J()-target1)/np.abs(target1):.3e}, ||residual||={r_max:.3e}")

tr = 1000
cross1 = np.zeros( (tr+1, 3) )
cross1[:-1, :] = s1.cross_section(0.50*2*np.pi/3., thetas=tr)
cross1[ -1, :] = cross1[0, :]
np.savetxt(f'cross_section_s1.txt', cross1)




s2 = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s2.set_dofs(dofs2[:-2])
iota0 = dofs2[-2]
G0 = dofs2[-1]
label2 = Volume(s2, stellarator)
bs2 = BiotSavart(stellarator.coils, stellarator.currents)
surf2 = BoozerSurface(bs2, s2, label2, target2)
res = surf2.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=300,iota=iota0,G=G0)
r, = boozer_surface_residual(s2, res['iota'], res['G'], bs2, derivatives=0)
r_max = np.linalg.norm(r, ord=np.inf)
print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={surf2.label.J():.3f}, vol_rel_error={np.abs(surf2.label.J()-target2)/np.abs(target2):.3e}, ||residual||={r_max:.3e}")

tr = 1000
cross2 = np.zeros( (tr+1, 3) )
cross2[:-1, :] = s2.cross_section(0.50*2*np.pi/3., thetas=tr)
cross2[ -1, :] = cross2[0, :]
np.savetxt(f'cross_section_s2.txt', cross2)




tr = 1000
crossExact = np.zeros((len(BoozerExact_list), tr+1, 3))
for idx, bs in enumerate(BoozerExact_list):
    s = bs.surface
    crossExact[idx, :-1, :] = s.cross_section(0.50*2*np.pi/3., thetas=tr)
    crossExact[idx,  -1, :] = crossExact[idx, 0, :]
for i in range(len(BoozerExact_list)):
    np.savetxt(f'cross_section_BoozerExact_{i}.txt', crossExact[i,:,:])

tr = 1000
crossLS = np.zeros((len(BoozerLS_list), tr+1, 3))
for idx, bs in enumerate(BoozerLS_list):
    s = bs.surface
    crossLS[idx, :-1, :] = s.cross_section(0.50*2*np.pi/3., thetas=tr)
    crossLS[idx,  -1, :] = crossLS[idx, 0, :]
for i in range(len(BoozerLS_list)):
    np.savetxt(f'cross_section_BoozerLS_{i}.txt', crossLS[i,:,:])






import matplotlib.pyplot as plt
p2 = np.loadtxt('./initial_poincare_Nt_coils=12/poincare2.txt', delimiter=',')
R = np.sqrt( p2[:,1]**2 + p2[:,2]**2)
Z = p2[:,3]
plt.scatter(R, Z, s=0.1, color=(0,0,1), alpha=0.1)
np.savetxt(f'p2.txt', p2[::20, 1:])


#for idx in range(crossExact.shape[0]):
#    R = np.sqrt(crossExact[idx,:,0]**2 + crossExact[idx,:,1]**2)
#    Z = crossExact[idx,:,2]
#    R = np.concatenate((R,[R[0]]))
#    Z = np.concatenate((Z,[Z[0]]))
#    if idx == 0:
#        plt.plot(R,Z, linewidth = 2, color = (1,0,0))
#    else:
#        plt.plot(R,Z, linewidth = 2, color = (0,0,0))





#for idx in range(crossLS.shape[0]):
#    R = np.sqrt(crossLS[idx,:,0]**2 + crossLS[idx,:,1]**2)
#    Z = crossLS[idx,:,2]
#    R = np.concatenate((R,[R[0]]))
#    Z = np.concatenate((Z,[Z[0]]))
#    if idx == 0:
#        plt.plot(R,Z, linewidth = 2, color = (1,0,0))
#    else:
#        plt.plot(R,Z, linewidth = 2, color = (0,0,0))


#plt.tight_layout()
#plt.show()
#plt.close()


