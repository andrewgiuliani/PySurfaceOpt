#!/usr/bin/env python3
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.field.coil import coils_via_symmetries
from simsopt.field.tracing import trace_particles_starting_on_surface, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo.curve import curves_to_vtk
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, ONE_EV
import simsoptpp as sopp
import pysurfaceopt as pys
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius, BoozerResidual, NonQuasiAxisymmetricRatio, Iotas, Volume, Area, Aspect_ratio


def minor_radius(surface):
    # see explanation for Surface.aspect_ratio in https://github.com/hiddenSymmetries/simsopt/blob/master/src/simsopt/geo/surface.py
    xyz = surface.gamma()
    x2y2 = xyz[:, :, 0]**2 + xyz[:, :, 1]**2
    dgamma1 = surface.gammadash1()
    dgamma2 = surface.gammadash2()

    # compute the average cross sectional area
    J = np.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
    J[:, :, 0, 0] = (xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0])/x2y2
    J[:, :, 0, 1] = (xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0])/x2y2
    J[:, :, 1, 0] = 0.
    J[:, :, 1, 1] = 1.

    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)

    dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
    mean_cross_sectional_area = np.abs(np.mean(np.sqrt(x2y2) * dZ_dtheta * detJ))/(2 * np.pi)

    R_minor = np.sqrt(mean_cross_sectional_area / np.pi)
    return R_minor

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank==0:
        print(f"There are {comm.size} ranks available.")
except ImportError:
    comm = None

import numpy as np
import time
import os
import logging
import sys

TMAX=2e-1

sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

length=18
assert length==18 or length==20 or length==22 or length==24
problem = pys.get_stageIII_problem(coilset='nine', length=length)
boozer_surface_list = [problem.boozer_surface_list[4], problem.boozer_surface_list[8]]
base_curves = problem._base_curves
base_currents = problem._base_currents
coils = problem.coils

bs_tf_list = [BiotSavart(coils) for bs in boozer_surface_list]
J_toroidal_flux = [ToroidalFlux(boozer_surface, bs_tf) for boozer_surface, bs_tf in zip(boozer_surface_list, bs_tf_list)]
J_aspect_ratio = [Aspect_ratio(boozer_surface.surface) for boozer_surface in boozer_surface_list]
tf_list=[abs(tf.J()) for tf in J_toroidal_flux]
ar_list=[ar.J() for ar in J_aspect_ratio]

print("compute the 0.25 surface")
# compute the s=0.25 surface
vol_dict={18:-0.14, 20:-0.1402, 22:-0.1402, 24:-0.1402}
boozer_surface_spawn = boozer_surface_list[0]
tf_spawn = J_toroidal_flux[0]
iota0=boozer_surface_spawn.res['iota']
G0=boozer_surface_spawn.res['G']
boozer_surface_spawn.targetlabel = vol_dict[length]
boozer_surface_spawn.recompute_bell()
tf_spawn.recompute_bell()
res = boozer_surface_spawn.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=iota0,G=G0)
tf_list=[abs(tf.J()) for tf in J_toroidal_flux]
ar_list=[ar.J() for ar in J_aspect_ratio]
max_tf = max(tf_list)
print("new max tf",max_tf)
print("tf:",tf_list/max_tf)
print("ar:",ar_list)
boozer_surface_outer=boozer_surface_list[1]

# convert these surfaces to full period surfaces
mpol = 10
ntor = 10
stellsym = True
nfp = 2
phis = np.linspace(0, 1, nfp*2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s_spawn = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s_outer = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

# rescale the minor radius of outermost surfaces to be 1.7
minor_radius_outer = minor_radius(boozer_surface_outer.surface)
s_spawn.x=boozer_surface_spawn.surface.x*1.7/minor_radius_outer
s_outer.x=boozer_surface_outer.surface.x*1.7/minor_radius_outer

comm.Barrier()
if comm.rank==0:
    s_spawn.to_vtk(OUT_DIR+"spawn_surface")
    s_outer.to_vtk(OUT_DIR+"outer_surface")
    print("Starting to particle trace...")    


nparticles = 2
degree = 3


"""
This examples demonstrate how to use SIMSOPT to compute guiding center
trajectories of particles
"""
for c in base_curves:
    c.x=c.x*1.7/minor_radius_outer

bs = BiotSavart(coils)
bs.set_points(s_outer.gamma().reshape(-1,3))
Bfield=bs.B()
modB = np.linalg.norm(Bfield, axis=-1)
avgB = np.mean(modB)
for curr in base_currents:
    curr.x=curr.x*5.9/avgB


bs = BiotSavart(coils)
bs.set_points(s_outer.gamma().reshape(-1,3))
Bfield=bs.B()
modB = np.linalg.norm(Bfield, axis=-1)
avgB = np.mean(modB)


if comm.rank==0:
    print("Mean(|B|) on outer surface =", avgB)
    print("Minor radius of outer surface =", minor_radius(s_outer))

sc_particle = SurfaceClassifier(s_outer, h=0.1, p=2)
n = 32
rs = np.linalg.norm(s_outer.gamma()[:, :, 0:2], axis=2)
zs = s_outer.gamma()[:, :, 2]

nfp=2
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/2, n*2)
zrange = (0, np.max(zs), n//2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=2, stellsym=True
)


def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    gc_tys, gc_phi_hits = trace_particles_starting_on_surface(
        s_spawn, bfield, nparticles, tmax=TMAX, seed=1, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
        Ekin=3.5e6*ONE_EV, umin=-1, umax=+1, comm=comm,
        phis=phis, tol=1e-10,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    t2 = time.time()
    print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)
    #particles_to_vtk(gc_tys, OUT_DIR + f'particles_{label}_{mode}_{comm.rank}')
    #plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{label}_loss_{comm.rank}.png', mark_lost=True)
    #plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{label}_{comm.rank}.png', mark_lost=False)
    return gc_tys
def compute_error_on_surface(s):
    bsh.set_points(s.gamma().reshape((-1, 3)))
    dBh = bsh.GradAbsB()
    Bh = bsh.B()
    bs.set_points(s.gamma().reshape((-1, 3)))
    dB = bs.GradAbsB()
    B = bs.B()
    print("Mean(|B|) on surface   %s" % np.mean(bs.AbsB()))
    print("B    errors on surface %s" % np.sort(np.abs(B-Bh).flatten()))
    print("∇|B| errors on surface %s" % np.sort(np.abs(dB-dBh).flatten()))

print("About to compute error")
compute_error_on_surface(s_spawn)
compute_error_on_surface(s_outer)
print("", flush=True)

paths_gc_h = trace_particles(bsh, 'bsh', 'gc_vac')

comm.Barrier()
if comm.rank==0:
    outname = f"losses_len{length}"
    np.save(OUT_DIR + outname, paths_gc_h)
    def get_lost_or_not(paths):
        return np.asarray([p[-1, 0] < TMAX-1e-15 for p in paths]).astype(int)
    print(f"{np.mean(get_lost_or_not(paths_gc_h))*100}%")

