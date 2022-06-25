import time
import numpy as np
import os
from mpi4py import MPI
from simsopt._core.graph_optimizable import Optimizable
import jax; jax.config.update('jax_platform_name', 'cpu')
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from pysurfaceopt.helpers import is_self_intersecting
from pysurfaceopt.curveobjectives import MeanSquareCurvature
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius, BoozerResidual, NonQuasiAxisymmetricRatio, Iotas, Volume, Area, Aspect_ratio
from pysurfaceopt.surfaceobjectives import boozer_surface_dlsqgrad_dcoils_vjp
from pysurfaceopt.surfaceobjectives import boozer_surface_dexactresidual_dcoils_dcurrents_vjp
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, ArclengthVariation, LpCurveCurvature 
from rich.console import Console
from rich.table import Column, Table



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
        self.bs_ratio_list = [BiotSavart(coils) for s in self.boozer_surface_list]
        self.bs_tf_list = [BiotSavart(coils) for s in self.boozer_surface_list]
        self.bs_boozer_residual_list = [BiotSavart(coils) for s in self.boozer_surface_list]
        
        self.J_coil_lengths    = [CurveLength(coil) for coil in self._base_curves]
        self.J_distance = MinimumDistance(self._all_curves, minimum_distance)
        self.J_major_radii = [MajorRadius(booz_surf) for booz_surf in self.boozer_surface_list]

        self.J_toroidal_flux = [ToroidalFlux(boozer_surface, bs_tf) for boozer_surface, bs_tf in zip(self.boozer_surface_list, self.bs_tf_list)]
        constraint_weights = [ booz_surf.res['constraint_weight'] if 'constraint_weight' in booz_surf.res else 0. for booz_surf in self.boozer_surface_list]
        self.J_boozer_residual = [BoozerResidual(boozer_surface, bs_boozer_residual, cw) for boozer_surface, bs_boozer_residual, cw in zip(boozer_surface_list, self.bs_boozer_residual_list, constraint_weights)]
        
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
        
        dependencies=[]
        dependencies+=self.J_curvature+self.J_msc+self.J_arclength+self.J_coil_lengths+[self.J_distance]
        dependencies+=self.J_nonQS_ratio+self.J_major_radii+self.J_toroidal_flux+self.J_boozer_residual+self.J_iotas
        
        if residual_weight is None:
            self.residual_weight = None
        else:
            self.residual_weight=residual_weight
        
        if lengthbound_threshold is None:
            self.lengthbound_threshold = None
        else:
            self.lengthbound_threshold = lengthbound_threshold

        if iotas_target is None:
            self.iotas_target = [None for i in range(len(self.boozer_surface_list))]
        else:
            self.iotas_target = iotas_target
        
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
            

        
        Optimizable.__init__(self, depends_on=dependencies)
        
        self.res_reference = None
        self.dres_referene = None
        self.update(verbose=True)
        
        
        self.x_reference = self.x
        self.res_reference = self.res
        self.dres_reference = self.dres

        self.output = output
        self.outdir="output"
        if outdir_append != "":
            self.outdir += "_" + outdir_append + "/"
        
        itarget_string = ','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_target) for i in rlist])
        ilb_string=','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_lb) for i in rlist])
        iub_string=','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_ub) for i in rlist])
        tftarget_string = ','.join('%.16e'%tf if tf is not None else "-" for tf in [i for rlist in MPI.COMM_WORLD.allgather(self.toroidal_flux_targets) for i in rlist])
        mRtarget_string = ','.join('%.16e'%m if m is not None else "-" for m in [i for rlist in MPI.COMM_WORLD.allgather(self.major_radii_targets) for i in rlist])

        if self.output and self.rank==0:
            os.makedirs(self.outdir, exist_ok=True)
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
                    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30, iota=iota0, G=G0)
                    res['solver'] = 'NEWTON'
                else:
                    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-13, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', 
                            hessian=False, weighting=boozer_surface.res['weighting'])
                    res['solver'] = 'LVM'
                    
                    # if close to minimum, try Newton
                    #if not res['success'] and np.linalg.norm(res['gradient'], ord=np.inf) < 1.:
                    #    boozer_surface.need_to_run_code = True
                    #    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=1e-13, maxiter=40, constraint_weight=100., iota=res['iota'], G=res['G'],
                    #            weighting=boozer_surface.res['weighting'])
                    #    res['solver'] = 'NEWTON'
                    
                    if np.linalg.norm(res['gradient'], ord=np.inf) < 1.:
                        # exactly constrained now
                        boozer_surface.need_to_run_code = True
                        res = boozer_surface.minimize_boozer_exact_constraints_newton(tol=1e-13, maxiter=30, iota=res['iota'], G=res['G'],
                                weighting=boozer_surface.res['weighting'])
                        res['solver'] = 'NEWTON_cons'
            except:
                boozer_surface.res['success']=False
        
        # one last try, let's do a continuation
        #for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
        #    if not boozer_surface.res['success']:
        #        boozer_surface.surface.set_dofs(reference_surface['dofs'])
        #        iota0 = reference_surface['iota']
        #        G0 = reference_surface['G']
        #        
        #        N_cont = 10
        #        print(f"Continuation on rank {self.rank}")
        #        x_old = self.x_reference.copy()
        #        x_new = self.x.copy()
        #        interp = np.linspace(0, 1, N_cont)
        #        for alpha in interp.tolist():
        #            self.x = alpha*x_new + (1-alpha)*x_old
        #             
        #            boozer_surface.need_to_run_code = True
        #            try: 
        #                if boozer_surface.res['type'] == 'exact':
        #                    res = boozer_surface.solve_residual_equation_exactly_newton( tol=1e-13, maxiter=30, iota=iota0, G=G0)
        #                    res['solver'] = 'NEWTON'
        #                else:
        #                    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-13, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', hessian=True)
        #                    res['solver'] = 'LVM'
        #                    
        #                    # if close to minimum, try Newton
        #                    if not res['success'] and np.linalg.norm(res['gradient'], ord=np.inf) < 1.:
        #                        boozer_surface.need_to_run_code = True
        #                        res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=1e-13, maxiter=40, constraint_weight=100., iota=res['iota'], G=res['G'])
        #                        res['solver'] = 'NEWTON'
        #                
        #                iota0=res['iota']
        #                G0=res['G']
        #            except:
        #                boozer_surface.res['success']=False
        #            print(f"rank={self.rank}, alpha={alpha}, solver={res['solver']}, success={res['success']}, iter={res['iter']}")
        #            
        #            if not boozer_surface.res['success']:
        #                self.x = x_new
        #                break # continuation failed

        success = True
        for bs in self.boozer_surface_list:
            success = success and bs.res['success']
        success_list = MPI.COMM_WORLD.allgather(success)

        res_list = [ {'solver':bs.res['solver'], 'type':bs.res['type'], 'success':bs.res['success'], 'iter':bs.res['iter'], 
                     'residual':bs.res['residual'], 'iota':bs.res['iota']} for bs in self.boozer_surface_list ]
        
        for res, bs in zip(res_list, self.boozer_surface_list):
            if 'labelerr' in bs.res:
                res['labelerr'] = bs.res['labelerr']
            if 'firstorderop' in bs.res:
                res['firstorderop'] = bs.res['firstorderop']

        if verbose: 
            res_list = [r for rlist in MPI.COMM_WORLD.allgather(res_list) for r in rlist]
            if self.rank==0:
                for res in res_list:
                    if res['type'] == 'exact':
                        print(f"{res['success']} - {res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e} ")
                    else:
                        print(f"{res['success']} - {res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['firstorderop'], ord=np.inf):.3e}, rel. label error: {res['labelerr']}")
                print("--------------------------------------------------------------------------------")
        
        if False in success_list:
            for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
                boozer_surface.surface.set_dofs(reference_surface['dofs'])
                iota0 = reference_surface['iota']
                G0 = reference_surface['G']
            if self.res_reference is None:
                print("failed to converge on init")
                quit()

            self.res = 2*self.res_reference
            self.dres = -self.dres_reference.copy() 
            if self.rank == 0:
                print("backtracking: failed surface solve")
                print("----------------------------------------------------------------------")
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
        if self.iotas_target_weight is not None:
            iotas_penalty = self.iotas_target_weight * sum([0.5 * (J2.J()-iotas_target)**2 if iotas_target is not None else 0. for J2, iotas_target in zip(self.J_iotas, self.iotas_target)])
            diotas_penalty = self.iotas_target_weight * sum([ (J2.J()-iotas_target)*J2.dJ(partials=True)(self) if iotas_target is not None else 0. for J2, iotas_target in zip(self.J_iotas, self.iotas_target)])
            iotas_list = MPI.COMM_WORLD.allgather(iotas_penalty)
            diotas_list = MPI.COMM_WORLD.allgather(diotas_penalty)
            res_dict[ 'iotas'] = sum(iotas_list)
            dres_dict['iotas'] = sum(diotas_list)
        
        if self.tf_weight is not None:
            tf_penalty  =  0.5 * self.tf_weight * sum( [ (J6.J() - l)**2 if l is not None else 0. for (J6, l) in zip(J_toroidal_flux, self.toroidal_flux_targets)])
            dtf_penalty =        self.tf_weight * sum( [ (J6.J() - l)*J6.dJ(partials=True)(self) if l is not None else 0. for (J6, l) in zip(J_toroidal_flux, self.toroidal_flux_targets)])
            tf_list  = MPI.COMM_WORLD.allgather(tf_penalty)
            dtf_list = MPI.COMM_WORLD.allgather(dtf_penalty)
            res_dict[ 'toroidal flux'] = sum(tf_list)
            dres_dict['toroidal flux'] = sum(dtf_list)
        
        if self.lengthbound_weight is not None: 
            res_dict[ 'lengthbound'] = self.lengthbound_weight * 0.5 * np.max([0, sum(J3.J() for J3 in J_coil_lengths)-self.lengthbound_threshold])**2
            dres_dict['lengthbound'] = self.lengthbound_weight * np.max([0, sum(J3.J() for J3 in J_coil_lengths)-self.lengthbound_threshold]) * sum(J3.dJ(partials=True)(self) for J3 in J_coil_lengths)
        
        if self.mr_weight is not None:
            mr  = 0.5 * self.mr_weight * sum(  (J3.J() - l)**2 if l is not None else 0. for (J3, l) in zip(J_major_radii, self.major_radii_targets))
            dmr = self.mr_weight * sum([ (J3.J()-l) * J3.dJ(partials=True)(self) if l is not None else 0. for (J3, l) in zip(J_major_radii, self.major_radii_targets)] ) 
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
            residual =  sum([self.residual_weight*res.J() for res in self.J_boozer_residual] )/self.Nsurfaces
            dresidual=  sum([self.residual_weight*res.dJ(partials=True)(self) for res in self.J_boozer_residual] )/self.Nsurfaces
            residual_list  = MPI.COMM_WORLD.allgather(residual)
            dresidual_list = MPI.COMM_WORLD.allgather(dresidual)
            res_dict[ 'BoozerResidual'] = sum(residual_list)
            dres_dict['BoozerResidual'] = sum(dresidual_list)
        
        if self.msc_weight is not None:
            res_dict[ 'msc']= self.msc_weight * sum([J13.J() for J13 in J_msc])
            dres_dict['msc']= self.msc_weight * sum([J13.dJ(partials=True)(self) for J13 in J_msc])
        
        if self.iotas_avg_target is not None:
            iotas_list = [r for rlist in MPI.COMM_WORLD.allgather([Jtemp.J() for Jtemp in J_iotas]) for r in rlist]
            diotas_list = [r for rlist in MPI.COMM_WORLD.allgather([Jtemp.dJ(partials=True)(self) for Jtemp in J_iotas]) for r in rlist]
            self.iotas_avg  = sum(iotas_list)/self.Nsurfaces
            diotas_avg = sum(diotas_list)/self.Nsurfaces
            res_dict['iotas avg'] = 0.5 * self.iotas_avg_weight * (self.iotas_avg - self.iotas_avg_target)**2
            dres_dict['iotas avg'] =      self.iotas_avg_weight * (self.iotas_avg - self.iotas_avg_target) * diotas_avg
        
        if self.iotas_bound_weight is not None:
            iotas_lb = self.iotas_bound_weight * sum([0.5 * max([iotas_lb-Jtemp.J(),0])**2 if iotas_lb is not None else 0. for Jtemp, iotas_lb in zip(self.J_iotas, self.iotas_lb)])
            diotas_lb= self.iotas_bound_weight * sum([     -max([iotas_lb-Jtemp.J() ,0]) * Jtemp.dJ(partials=True)(self) for Jtemp, iotas_lb in zip(self.J_iotas, self.iotas_lb)])
            iotas_lb_list = MPI.COMM_WORLD.allgather(iotas_lb)
            diotas_lb_list= MPI.COMM_WORLD.allgather(diotas_lb)
            res_dict['iotas_lb'] = sum(iotas_lb_list)
            dres_dict['iotas_lb'] = sum(diotas_lb_list)
        
        if self.iotas_bound_weight is not None:
            iotas_ub = self.iotas_bound_weight * sum([0.5 * max([Jtemp.J()-iotas_ub,0])**2 if iotas_ub is not None else 0. for Jtemp, iotas_ub in zip(self.J_iotas, self.iotas_ub)])
            diotas_ub= self.iotas_bound_weight * sum([ max([Jtemp.J()-iotas_ub ,0]) * Jtemp.dJ(partials=True)(self) for Jtemp, iotas_ub in zip(self.J_iotas, self.iotas_ub)])
            iotas_ub_list = MPI.COMM_WORLD.allgather(iotas_ub)
            diotas_ub_list= MPI.COMM_WORLD.allgather(diotas_ub)
            res_dict['iotas_ub'] = sum(iotas_ub_list)
            dres_dict['iotas_ub'] = sum(diotas_ub_list)

        self.res =  sum(res_dict.values())
        self.dres = sum(dres_dict.values())
        
        self.res_dict=res_dict
        self.dres_dict=dres_dict

    def callback(self, x):
        assert np.allclose(self.x, x, rtol=0, atol=0)
        
        # check self.x is the same at all ranks
        all_x = MPI.COMM_WORLD.allgather(self.x.copy())
        for xx in all_x:
            assert np.allclose(self.x, xx, rtol=0, atol=0)

        # update reference dofs used as initial guesses during line search.
        self.boozer_surface_reference = [{"dofs": boozer_surface.surface.get_dofs(),
                                         "iota": boozer_surface.res["iota"],
                                          "G": boozer_surface.res["G"]} for boozer_surface in self.boozer_surface_list]
        # update reference res, and dres
        self.x_reference = x.copy()
        self.res_reference = self.res
        self.dres_reference = self.dres.copy()


        self.iter+=1

        xs_list = [bs.surface.cross_section(0., thetas=np.linspace(0, 1, 100, endpoint=False)) for bs in self.boozer_surface_list]
        si_list = [is_self_intersecting(xs) for xs in xs_list]
        all_si_list = [bval for s_list in MPI.COMM_WORLD.allgather(si_list) for bval in s_list]
        if self.rank==0:
            if True in all_si_list:
                bool_string = " ".join(['True' if b else 'False' for b in all_si_list])
                print("Self-intersections " + bool_string)

        def compute_non_quasisymmetry_L2(in_surface):
            bs = BiotSavart(self.coils)
            phis = np.linspace(0, 1/in_surface.nfp, 100, endpoint=False)
            thetas = np.linspace(0, 1., 100, endpoint=False)
            s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
            s.set_dofs(in_surface.get_dofs())

            x = s.gamma()
            B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
            mod_B = np.linalg.norm(B, axis=2)
            n = np.linalg.norm(s.normal(), axis=2)
            mean_phi_mod_B = np.mean(mod_B*n, axis=0)/np.mean(n, axis=0)
            mod_B_QS = mean_phi_mod_B[None, :]
            mod_B_non_QS = mod_B - mod_B_QS
            non_qs = np.mean(mod_B_non_QS**2 * n)**0.5
            qs = np.mean(mod_B_QS**2 * n)**0.5
            return non_qs, qs

        ratio = []
        for booz_surf in self.boozer_surface_list:
            non_qs, qs = compute_non_quasisymmetry_L2(booz_surf.surface)
            ratio.append(non_qs/qs)

        res_list = [ {'type':bs.res['type'], 'success':bs.res['success'], 'iter':bs.res['iter'], 'residual':bs.res['residual']} for bs in self.boozer_surface_list ]

        for res, bs in zip(res_list, self.boozer_surface_list):
            if 'labelerr' in bs.res:
                res['labelerr'] = bs.res['labelerr']
            if 'firstorderop' in bs.res:
                res['firstorderop'] = bs.res['firstorderop']

        
        res_list = [r for rlist in MPI.COMM_WORLD.allgather(res_list) for r in rlist]

        iotas = [ abs(res['iota']) for res in self.boozer_surface_reference ]
        volumes = [abs(Volume(bs.surface).J()) for bs in self.boozer_surface_list]
        areas = [abs(Area(bs.surface).J()) for bs in self.boozer_surface_list]
        aspect_ratios = [Aspect_ratio(bs.surface).J() for bs in self.boozer_surface_list]
        mR = [R.J() for R in self.J_major_radii]
        tf = [tflux.J() for tflux in self.J_toroidal_flux]
        boozerRes = [Jbr.J() for Jbr in self.J_boozer_residual]

        
        other_char = {}
        other_char['||non qs||_2 / ||qs||_2'] = ' '.join('%.6e'%r for r in [i for d in MPI.COMM_WORLD.allgather(ratio) for i in d] )
        other_char['boozer residual'] = ' '.join('%.6e'%ar for ar in [i for d in MPI.COMM_WORLD.allgather(boozerRes) for i in d])
        other_char['total length'] = f'{sum([J.J() for J in self.J_coil_lengths]):.6e}'
        other_char['coil lengths'] = ' '.join([f"{J.J():.6e}" for J in self.J_coil_lengths])
        other_char['min_arc_length'] = f"{min([np.min(np.abs(coil.incremental_arclength())) for coil in self._base_curves]):.6e}"
        other_char['minimum distance'] = f"{self.J_distance.shortest_distance():.6e}"
        other_char['curvature'] = " ".join([f"{np.max(c.kappa()):.6e}" for c in self._base_curves])
        other_char['msc'] = " ".join([f"{Jmsc.msc():.6e}" for Jmsc in self.J_msc])
        other_char['iotas'] = ' '.join('%.6e'%i for i in [i for d in MPI.COMM_WORLD.allgather(iotas) for i in d] )
        if self.iotas_avg_target is not None:
            other_char["iotas avg"] = f"{abs(self.iotas_avg):.6e}"
            other_char["iotas_avg_target"] = f"{abs(self.iotas_avg_target):.6e}"
        if self.iotas_target is not None:
            other_char["iotas_target"] = ' '.join('%.6e'%abs(i) if i is not None   else "------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.iotas_target) for i in d] )
        if self.iotas_bound_weight is not None:
            other_char["iotas_lb"] = ' '.join('%.6e'%abs(i) if i is not None  else "------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.iotas_lb) for i in d] )
        if self.iotas_bound_weight is not None:
            other_char["iotas_ub"] = ' '.join('%.6e'%abs(i) if i is not None  else "------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.iotas_ub) for i in d])
        other_char['aspect ratios'] = ' '.join('%.6e'%abs(ar) for ar in [i for d in MPI.COMM_WORLD.allgather(aspect_ratios) for i in d])
        other_char['volumes'] = ' '.join('%.6e'%abs(v) for v in [i for d in MPI.COMM_WORLD.allgather(volumes) for i in d])
        other_char['areas'] = ' '.join('%.6e'%abs(a) for a in [i for d in MPI.COMM_WORLD.allgather(areas) for i in d])
        other_char['major radii'] = ' '.join('%.6e'%mr for mr  in [i for d in MPI.COMM_WORLD.allgather(mR) for i in d])
        if self.mr_weight is not None:
            other_char["major radii targets"] = ' '.join('%.6e'%i if i is not None  else "------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.major_radii_targets) for i in d])
        other_char['toroidal flux'] = ' '.join('%.6e'%mr for mr  in [i for d in MPI.COMM_WORLD.allgather(tf) for i in d])
        if self.tf_weight is not None:
            other_char["toroidal flux targets"] = ' '.join('%.6e'%i if i is not None  else "------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.toroidal_flux_targets) for i in d])

        if self.rank == 0:
            print(f"Iteration {self.iter}")
            print(f"Objective value:             {self.res:.6e}")
            print(f"Gradient:                    {np.linalg.norm(self.dres, ord=np.inf):.6e}")
            console = Console(width=200)
            table1 = Table(expand=True, show_header=False)
            table1.add_row(*[f"{v}" for v in self.res_dict.keys()])
            table1.add_row(*[f"{v:.6e}" for v in self.res_dict.values()])
            console.print(table1)

            table2 = Table(expand=True, show_header=False) 
            for k in other_char.keys():
                table2.add_row(k, other_char[k])
            console.print(table2)

            for res in res_list:
                if res['type'] == 'exact':
                    print(f"{res['success']} - iter={res['iter']}, success={res['success']}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
                else:
                    print(f"{res['success']} - iter={res['iter']}, success={res['success']}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||firstorderopt||_inf = {np.linalg.norm(res['firstorderop'], ord=np.inf):.3e}, rel. label error: {res['labelerr']}")
 
            print("################################################################################")
            if self.output:
                np.savetxt(self.outdir + f"x_{self.iter}.txt", self.x.reshape((1,-1)))

    def recompute_bell(self, parent=None):
        self.res=None
        self.dres=None
    def J(self, verbose=False):
        if self.rank == 0:
            np.savetxt(self.outdir+f"descent_direction_{self.iter}_{time.time()}.txt", self.x-self.x_reference)
        
        if self.res is None:
            self.update(verbose=verbose)
        
        return self.res
    def dJ(self, verbose=False):
        if self.dres is None:
            self.update(verbose=verbose)
        return self.dres
