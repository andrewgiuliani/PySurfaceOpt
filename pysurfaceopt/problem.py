import numpy as np
import os
from mpi4py import MPI
from simsopt._core.graph_optimizable import Optimizable
import jax; jax.config.update('jax_platform_name', 'cpu')
from simsopt.field.biotsavart import BiotSavart
from pysurfaceopt.curveobjectives import MeanSquareCurvature
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius, BoozerResidual, NonQuasiAxisymmetricRatio, Iotas
from pysurfaceopt.surfaceobjectives import boozer_surface_dlsqgrad_dcoils_vjp
from pysurfaceopt.surfaceobjectives import boozer_surface_dexactresidual_dcoils_dcurrents_vjp
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, ArclengthVariation, LpCurveCurvature 


class SurfaceProblem(Optimizable):
    def __init__(self, boozer_surface_list, base_curves, base_currents, coils,
                 iotas_target=None, iotas_lb=None, iotas_ub=None, iotas_avg_target=None,
                 major_radii_targets=None, 
                 toroidal_flux_targets=None, lengthbound_threshold=None,
                 kappa_max=None, msc_max=None, minimum_distance=None,
                 iotas_target_weight=None, iotas_bound_weight=None, iotas_avg_weight=None, mr_weight=None, tf_weight=None,
                 distance_weight=None, arclength_weight=None, curvature_weight=None, lengthbound_weight=None, msc_weight=None,
                 residual_weight=None, 
                 outdir_append="", output=True, rank=0):
        
        self.rank = rank
        self.boozer_surface_list = boozer_surface_list
        self._base_curves   = base_curves
        self._all_curves = [c.curve for c in coils]
        self._base_currents = base_currents
        self.coils = coils

        # communicate number of total surfaces across ranks
        Ns_rank = MPI.COMM_WORLD.allgather(len(self.boozer_surface_list))
        self.Nsurfaces = sum(Ns_rank)


        # reference dofs used as initial guesses during line search.
        self.boozer_surface_reference = [{"dofs": boozer_surface.surface.get_dofs(),
                                         "iota": boozer_surface.res["iota"],
                                          "G": boozer_surface.res["G"]} for boozer_surface in self.boozer_surface_list]
        
        self.iter = 0
        self.J_distance = MinimumDistance(self._all_curves, minimum_distance)
        self.J_major_radii = [MajorRadius(booz_surf) for booz_surf in self.boozer_surface_list]
        self.bs_ratio_list = [BiotSavart(coils) for s in self.boozer_surface_list]
        self.bs_tf_list = [BiotSavart(coils) for s in self.boozer_surface_list]
        self.bs_boozer_residual_list = [BiotSavart(coils) for s in self.boozer_surface_list]
        self.J_toroidal_flux = [ToroidalFlux(boozer_surface, bs_tf) for boozer_surface, bs_tf in zip(self.boozer_surface_list, self.bs_tf_list)]
        constraint_weights = [ booz_surf.res['constraint_weight'] if 'constraint_weight' in booz_surf.res else 0. for booz_surf in self.boozer_surface_list]
        self.J_boozer_residual = [BoozerResidual(boozer_surface, bs_boozer_residual, cw) for boozer_surface, bs_boozer_residual, cw in zip(boozer_surface_list, self.bs_boozer_residual_list, constraint_weights)]
        self.J_coil_lengths    = [CurveLength(coil) for coil in self._base_curves]
        
        self.msc_weight = msc_weight
        self.distance_weight = distance_weight
        self.arclength_weight = arclength_weight
        self.curvature_weight = curvature_weight
        self.lengthbound_weight = lengthbound_weight
        self.tf_weight=tf_weight
        self.iotas_target_weight=iotas_target_weight
        self.iotas_bound_weight=iotas_bound_weight
        self.iotas_avg_weight=iotas_avg_weight
        self.mr_weight=mr_weight
        self.J_curvature = [LpCurveCurvature(c, 2, threshold=kappa_max) for c in self._base_curves]
        self.J_msc = [MeanSquareCurvature(c, msc_max) for c in self._base_curves]
        self.J_arclength = [ArclengthVariation(curve) for curve in self._base_curves]
        self.J_nonQS_ratio = [NonQuasiAxisymmetricRatio(boozer_surface, bs) for boozer_surface, bs in zip(self.boozer_surface_list, self.bs_ratio_list)]
        self.J_iotas = [Iotas(boozer_surface) for boozer_surface in self.boozer_surface_list]
        
        self.msc_max = msc_max
        self.kappa_max = kappa_max

        dependencies = []
        
        dependencies+=self.J_nonQS_ratio
        if residual_weight is None:
            self.residual_weight = [None for booz_s in self.boozer_surface_list]
        else:
            self.residual_weight=residual_weight
            dependencies+=self.J_boozer_residual

        if lengthbound_threshold is None:
            self.lengthbound_threshold = None
        else:
            self.lengthbound_threshold = lengthbound_threshold
            dependencies+=self.J_coil_lengths

        if iotas_target is None:
            self.iotas_target = None
        else:
            self.iotas_target = iotas_target
            dependencies+=self.J_iotas
        
        if iotas_lb is None:
            self.iotas_lb = [None for i in range(len(self.boozer_surface_list))]
        else:
            self.iotas_lb = iotas_lb

        if iotas_ub is None:
            self.iotas_ub = [None for i in range(len(self.boozer_surface_list))]
        else:
            self.iotas_ub = iotas_ub

        if iotas_avg_target is None:
            self.iotas_avg_target = None
        else:
            self.iotas_avg_target = iotas_avg_target

        if major_radii_targets is None:
            self.major_radii_targets = [None for i in range(len(self.boozer_surface_list))]
        else:
            self.major_radii_targets = major_radii_targets
        
        if toroidal_flux_targets is None:
            self.toroidal_flux_targets = [None for i in range(len(self.boozer_surface_list))]
        else:
            self.toroidal_flux_targets = toroidal_flux_targets
        
        self.current_fak = 1./(4 * np.pi * 1e-7)
        Optimizable.__init__(self, depends_on=dependencies)
        self.update()
        self.x_reference = self.x
        self.res_reference = self.res
        self.dres_reference = self.dres

        self.output = output
        self.outdir = f"output" + outdir_append + "/"

        MPI.COMM_WORLD.barrier()
        if self.output and self.rank==0:
            os.makedirs(self.outdir, exist_ok=True)
            itarget_string = ','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_target) for i in rlist])
            ilb_string=','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_lb) for i in rlist])
            iub_string=','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_ub) for i in rlist])
            tftarget_string = ','.join('%.16e'%tf if tf is not None else "-" for tf in [i for rlist in MPI.COMM_WORLD.allgather(self.toroidal_flux_targets) for i in rlist])
            mRtarget_string = ','.join('%.16e'%m if m is not None else "-" for m in [i for rlist in MPI.COMM_WORLD.allgather(self.major_radii_targets) for i in rlist])

            out_targets = open(self.outdir + "out_targets.txt", "w")
            out_targets.write("iotas_targets " + itarget_string+"\n")
            out_targets.write("iotas_targets_weight " + str(self.iotas_target_weight) +"\n")
            out_targets.write("tf_targets " + tftarget_string+"\n")
            out_targets.write("tf_weight " + str(self.tf_weight)+"\n")
            out_targets.write("mR_targets " + mRtarget_string+"\n")
            out_targets.write("mr_weight " + str(self.mr_weight)+"\n")
            out_targets.write("iotas_lb " + ilb_string + "\n")
            out_targets.write("iotas_ub " + iub_string + "\n")
            out_targets.write("iotas_bound_weight " + str(self.iotas_bound_weight)+"\n")
            out_targets.write("avg_iotas_weight " + str(self.iotas_avg_weight)+"\n")
            out_targets.write("avg_iotas_target " + str(self.iotas_avg_target)+"\n")
            out_targets.write("kappa_max " + str(self.kappa_max)+"\n")
            out_targets.write("curvature_weight " + str(self.curvature_weight)+"\n")
            out_targets.write("msc_max " + str(self.msc_max)+"\n")
            out_targets.write("msc weight " + str(self.msc_weight)+"\n")
            out_targets.write("alen weight " + str(self.arclength_weight)+"\n")
            out_targets.write("distance weight " + str(self.distance_weight)+"\n")

            out_targets.close()
    
    def update(self, verbose=False):
        # compute surfaces
        for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
            boozer_surface.surface.set_dofs(reference_surface['dofs'])
            iota0 = reference_surface['iota']
            G0 = reference_surface['G']
            try: 
                if boozer_surface.res['type'] == 'exact':
                    res = boozer_surface.solve_residual_equation_exactly_newton( tol=1e-13, maxiter=30, iota=iota0, G=G0)
                    res['solver'] = 'NEWTON'
                else:
                    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                    res['solver'] = 'LS'
                    if not res['success']:
                        res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=5e-10, maxiter=30, constraint_weight=100., iota=res['iota'], G=res['G'])
                        res['solver'] = 'NEWTON'
            except:
               boozer_surface.res['success']=False
        

        success = True
        for bs in self.boozer_surface_list:
            success = success and bs.res['success']
        success_list = MPI.COMM_WORLD.allgather(success)

        res_list = [ {'solver':bs.res['solver'], 'type':bs.res['type'], 'success':bs.res['success'], 'iter':bs.res['iter'], 'residual':bs.res['residual'], 'iota':bs.res['iota']} for bs in self.boozer_surface_list ]
        for res, bs in zip(res_list, self.boozer_surface_list):
            if bs.res['type'] != 'exact':
                res['gradient'] = bs.res['gradient']
        

        if verbose: 
            res_list = [r for rlist in MPI.COMM_WORLD.allgather(res_list) for r in rlist]
            for res in res_list:
                if res['type'] == 'exact':
                    print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e} ")
                else:
                    print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
        
        if False in success_list:
            print("backtracking: failed surface solve")
            for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
                boozer_surface.surface.set_dofs(reference_surface['dofs'])
                iota0 = reference_surface['iota']
                G0 = reference_surface['G']
            if self.res_reference is None:
                print("failed to converge on init")
                quit()

            self.res = 2*self.res_reference
            self.dres = -self.dres_reference.copy() 
            print("--------------------------------------------------------------------------------")
            return


        J_iotas = self.J_iotas
        J_coil_lengths    = self.J_coil_lengths
        J_major_radii    = self.J_major_radii
        J_distance = self.J_distance
        J_toroidal_flux = self.J_toroidal_flux
        J_arclength = self.J_arclength
        J_curvature = self.J_curvature
        J_msc = self.J_msc
        J_nonQS_ratio = self.J_nonQS_ratio
        
        res_dict={}
        dres_dict={}
        

        ratio_list  = [r for rlist in MPI.COMM_WORLD.allgather([Jtemp.J() for Jtemp in J_nonQS_ratio]) for r in rlist]
        dratio_list = [r for rlist in MPI.COMM_WORLD.allgather([Jtemp.dJ(partials=True)(self) for Jtemp in J_nonQS_ratio]) for r in rlist]

        res_dict['ratio']  = sum(ratio_list)/self.Nsurfaces
        dres_dict['ratio'] = sum(dratio_list)/self.Nsurfaces
        if self.iotas_target is not None:
            iotas_penalty = self.iotas_target_weight * sum([0.5 * (J2.J()-iotas_target)**2 if iotas_target is not None else 0. for J2, iotas_target in zip(self.J_iotas, self.iotas_target)])
            diotas_penalty = self.iotas_target_weight * sum([ (J2.J()-iotas_target)*J2.dJ(partials=True)(self) if iotas_target is not None else 0. for J2, iotas_target in zip(self.J_iotas, self.iotas_target)])
            iotas_list = MPI.COMM_WORLD.allgather(iotas_penalty)
            diotas_list = MPI.COMM_WORLD.allgather(diotas_penalty)
            res_dict[ 'iotas'] = sum(iotas_list)
            dres_dict['iotas'] = sum(diotas_list)

        if self.lengthbound_threshold is not None: 
            res_dict[ 'lengthbound'] = self.lengthbound_weight * 0.5 * np.max([0, sum(J3.J() for J3 in J_coil_lengths)-self.lengthbound_threshold])**2
            dres_dict['lengthbound'] = self.lengthbound_weight * np.max([0, sum(J3.J() for J3 in J_coil_lengths)-self.lengthbound_threshold]) * sum(J3.dJ(partials=True)(self) for J3 in J_coil_lengths)
        if self.mr_weight is not None:
            mr  = 0.5 * self.mr_weight * sum(  (J3.J() - l)**2 if l is not None else 0. for (J3, l) in zip(J_major_radii, self.major_radii_targets))
            dmr = self.mr_weight * sum([ (J3.J()-l) * J3.dJ(partials=True)(self) if l is not None else 0. for (J3, l) in zip(J_major_radii, self.major_radii_targets)] ) * self.current_fak
            mr_list  = MPI.COMM_WORLD.allgather(mr)
            dmr_list = MPI.COMM_WORLD.allgather(dmr)
            res_dict[ 'mr'] = sum(mr_list)
            dres_dict['mr'] = sum(dmr_list)
        if self.distance_weight is not None:
            res_dict[ 'distance']= self.distance_weight * J_distance.J()
            dres_dict['distance']= self.distance_weight * J_distance.dJ(partials=True)(self)
        if self.arclength_weight is not None:
            res_dict[ 'alen']= self.arclength_weight * sum([J7.J() for J7 in J_arclength])
            dres_dict['alen']= self.arclength_weight * sum([J7.dJ(partials=True)(self) for J7 in J_arclength])
        if self.curvature_weight is not None:
            res_dict[ 'curvature'] = self.curvature_weight * sum([J8.J() for J8 in J_curvature])
            dres_dict['curvature'] = self.curvature_weight * sum([J8.dJ(partials=True)(self) for J8 in J_curvature])
        if self.residual_weight is not None:
            residual =  sum([res_weight*res.J() if res_weight is not None else 0. for res_weight, res in zip(self.residual_weight, self.J_boozer_residual)] )
            dresidual=  sum([res_weight*res.dJ(partial=True)(self) if res_weight is not None else 0. for res_weight, res in zip(self.residual_weight, self.J_boozer_residual)] )
            residual_list  = MPI.COMM_WORLD.allgather(residual)
            dresidual_list = MPI.COMM_WORLD.allgather(dresidual)
            res_dict[ 'residual'] = sum(residual_list)
            dres_dict['residual'] = sum(dresidual_list)
        if self.msc_weight is not None:
            res_dict[ 'msc']= self.msc_weight * sum([J13.J() for J13 in J_msc])
            dres_dict['msc']= self.msc_weight * sum([J13.dJ(partials=True)(self) for J13 in J_msc])
        if self.iotas_avg_target is not None:
            iotas_list = [r for rlist in MPI.COMM_WORLD.allgather([Jtemp.J() for Jtemp in J_iotas]) for r in rlist]
            diotas_list = [r for rlist in MPI.COMM_WORLD.allgather([Jtemp.dJ(partials=True)(self) for Jtemp in J_iotas]) for r in rlist]
            self.iotas_avg  = sum(iotas_list)/self.Nsurfaces
            diotas_avg = sum(diotas_list)/self.Nsurfaces
            res_dict['iotas'] = 0.5 * self.iotas_avg_weight * ( self.iotas_avg - self.iotas_avg_target)**2
            dres_dict['iotas'] =      self.iotas_avg_weight * ( self.iotas_avg - self.iotas_avg_target) * diotas_avg

        self.res =  sum(res_dict.values())
        self.dres = sum(dres_dict.values())
    
    def recompute_bell(self, parent=None):
        self.update()
    def J(self, verbose=False):
        return self.res
    def dJ(self, verbose=False):
        return self.dres
