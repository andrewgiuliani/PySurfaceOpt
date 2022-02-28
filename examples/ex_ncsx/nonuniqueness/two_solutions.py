#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.util.zoo import get_ncsx_data
from simsopt.field.coil import ScaledCurrent, Current, coils_via_symmetries

PPP=20
Nt_coils=12
base_curves, base_currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
base_currents = [Current(curr.x*4 * np.pi * 1e-7) for curr in base_currents]
base_currents = [ScaledCurrent(curr, 1/(4 * np.pi * 1e-7)) for curr in base_currents]

nfp = ma.nfp
stellsym = True
coils = coils_via_symmetries(base_curves, base_currents, ma.nfp, stellsym)



dofs1 = np.loadtxt('BoozerExact_surface1.txt')
dofs2 = np.loadtxt('BoozerExact_surface2.txt')

stellsym=True
nfp = 3
ntor = 11
mpol = 11
nquadphi = ntor+1
nquadtheta = 2*mpol+1
phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

s1 = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s1.x=dofs1[:-2]
label1 = pys.Volume(s1)
bs1 = BiotSavart(coils)
target1=-1
surf1 = BoozerSurface(bs1, s1, label1, target1)

iota0=dofs1[-2]
G0=dofs1[-1]
res = surf1.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=300,iota=iota0,G=G0)
r, = pys.boozer_surface_residual(s1, res['iota'], res['G'], bs1, derivatives=0)
r_max = np.linalg.norm(r, ord=np.inf)
print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={surf1.label.J():.3f}, vol_rel_error={np.abs(surf1.label.J()-target1)/np.abs(target1):.3e}, ||residual||={r_max:.3e}")

bs2= BiotSavart(coils)
s2 = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s2.x=dofs2[:-2]
label2 = pys.Volume(s2)
bs1 = BiotSavart(coils)
target2=-1
surf2 = BoozerSurface(bs2, s2, label2, target2)

iota0=dofs2[-2]
G0=dofs2[-1]
res = surf2.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=300,iota=iota0,G=G0)
r, = pys.boozer_surface_residual(s2, res['iota'], res['G'], bs1, derivatives=0)
r_max = np.linalg.norm(r, ord=np.inf)
print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={surf2.label.J():.3f}, vol_rel_error={np.abs(surf2.label.J()-target2)/np.abs(target2):.3e}, ||residual||={r_max:.3e}")



cs1=s1.cross_section(0,thetas=1000)
cs2=s2.cross_section(0,thetas=1000)
def wrap(cs):
    return np.concatenate((cs, cs[0,:][None,:]), axis=0)
import matplotlib.pyplot as plt
plt.plot(np.sqrt(cs1[:,0]**2+ cs1[:,1]**2), cs1[:,2])
plt.plot(np.sqrt(cs2[:,0]**2+ cs2[:,1]**2), cs2[:,2])
plt.show()
