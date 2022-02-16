import numpy as np
import os
from mpi4py import MPI
import jax; jax.config.update('jax_platform_name', 'cpu')
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, UniformArclength, LpCurveCurvature, QuadraticCurveLength, MeanSquareCurvature
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricRatio,NonQuasiAxisymmetricComponent, Iotas, Area, Volume, Aspect_ratio, ToroidalFlux, boozer_surface_residual, MajorRadius, AreaElement, AreaPenalty, BoozerResidual
from simsopt.geo.coilcollection import CoilCollection
from .logging import info
from .logging import set_file_logger

def is_self_intersecting(cs):
    """
    This function takes as input a cross section, represented as a polygon.
    """
    R = np.sqrt( cs[:,0]**2 + cs[:,1]**2)
    Z = cs[:, 2]

    from ground.base import get_context
    context = get_context()
    Point, Contour = context.point_cls, context.contour_cls
    contour = Contour([ Point(R[i], Z[i]) for i in range(cs.shape[0]) ])
    from bentley_ottmann.planar import contour_self_intersects
    return contour_self_intersects(contour)

class SurfaceProblem(object):
    def __init__(self, boozer_surface_list, stellarator,
                 iotas_target=None, iotas_lb=None, iotas_ub=None, iotas_avg_target=None,
                 major_radii_targets=None, 
                 toroidal_flux_targets=None, lengthbound_threshold=None,
                 kappa_max=5, msc_max=5, minimum_distance=0.15,
                 initial_areas=None,
                 iotas_target_weight=0., iotas_bound_weight=0., iotas_avg_weight=0., mr_weight=0., tf_weight=0.,
                 distance_weight=0., arclength_weight=0, curvature_weight=0, lengthbound_weight=0., msc_weight=0.,
                 residual_weight=None, 
                 outdir_append="", output=True, rank=0):
        
        self.rank = rank
        self.boozer_surface_list = boozer_surface_list
        self.stellarator = stellarator
        

        values = MPI.COMM_WORLD.allgather(len(self.boozer_surface_list))
        self.Nsurfaces = sum(values)


        # reference dofs used as initial guesses during line search.
        self.boozer_surface_reference = [{"dofs": boozer_surface.surface.get_dofs(),
                                         "iota": boozer_surface.res["iota"],
                                          "G": boozer_surface.res["G"]} for boozer_surface in self.boozer_surface_list]
        
        self.iter = 0
        self.J_distance = MinimumDistance(stellarator.coils, minimum_distance)
        self.J_major_radii = [MajorRadius(booz_surf, self.stellarator) for booz_surf in self.boozer_surface_list]
        self.bs_nonqs_list = [BiotSavart(stellarator.coils, stellarator.currents) for s in self.boozer_surface_list]
        self.bs_ratio_list = [BiotSavart(stellarator.coils, stellarator.currents) for s in self.boozer_surface_list]
        self.bs_tf_list = [BiotSavart(stellarator.coils, stellarator.currents) for s in self.boozer_surface_list]
        self.bs_boozer_residual_list = [BiotSavart(stellarator.coils, stellarator.currents) for s in self.boozer_surface_list]
        self.J_toroidal_flux = [ToroidalFlux(boozer_surface.surface, bs_tf, stellarator, boozer_surface=boozer_surface) for boozer_surface, bs_tf in zip(self.boozer_surface_list, self.bs_tf_list)]
        self.J_area = [AreaPenalty(boozer_surface) for boozer_surface in boozer_surface_list]
        self.J_boozer_residual = [BoozerResidual(boozer_surface, bs_boozer_residual, 100.) for boozer_surface, bs_boozer_residual in zip(boozer_surface_list, self.bs_boozer_residual_list)]
        self.initial_areas = initial_areas 
        
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

        if residual_weight is None:
            self.residual_weight = [None for booz_s in self.boozer_surface_list]
        else:
            self.residual_weight=residual_weight

        self.J_coil_lengths    = [CurveLength(coil) for coil in stellarator._base_coils]
        if lengthbound_threshold is not None:
            self.lengthbound_threshold = lengthbound_threshold
        else:
            self.lengthbound_threshold = sum([J.J() for J in self.J_coil_lengths])
        self.J_lengthbound_penalty = QuadraticCurveLength(self.J_coil_lengths, self.lengthbound_threshold)

        if iotas_target is None:
            self.iotas_target = None
        else:
            self.iotas_target = iotas_target
        
        if iotas_lb is None:
            self.iotas_lb = None
        else:
            self.iotas_lb = iotas_lb

        if iotas_ub is None:
            self.iotas_ub = None
        else:
            self.iotas_ub = iotas_ub

        if iotas_avg_target is None:
            self.iotas_avg_target = None
        else:
            self.iotas_avg_target = iotas_avg_target

        if major_radii_targets is None:
            self.major_radii_targets = [J.J() for J in self.J_major_radii]
        else:
            self.major_radii_targets = major_radii_targets
        
        if toroidal_flux_targets is None:
            self.toroidal_flux_targets = [J.J() for J in self.J_toroidal_flux]
        else:
            self.toroidal_flux_targets = toroidal_flux_targets
        
        if initial_areas is None:
            self.initial_areas = None
        else:
            self.initial_areas = initial_areas
        
        self.msc_max = msc_max
        self.kappa_max = kappa_max
        self.J_curvature = [LpCurveCurvature(c, 2, desired_length=2*np.pi/self.kappa_max) for c in stellarator._base_coils]
        self.J_msc = [MeanSquareCurvature(c, self.msc_max) for c in stellarator._base_coils]
        self.J_arclength = [UniformArclength(curve) for curve in stellarator._base_coils]
        self.J_nonQS = [NonQuasiAxisymmetricComponent(boozer_surface, bs) for boozer_surface, bs in zip(self.boozer_surface_list, self.bs_nonqs_list)]
        self.J_nonQS_ratio = [NonQuasiAxisymmetricRatio(boozer_surface, bs) for boozer_surface, bs in zip(self.boozer_surface_list, self.bs_ratio_list)]
        self.J_iotas = [Iotas(boozer_surface) for boozer_surface in self.boozer_surface_list]
        

        self.current_fak = 1./(4 * np.pi * 1e-7)
        self.idx_curr = np.arange(0,len(stellarator.get_currents()))
        self.idx_coil = np.arange(self.idx_curr[-1]+1, self.idx_curr[-1]+1 + len(stellarator.get_dofs()))
        self.x = np.concatenate([stellarator.get_currents()/self.current_fak, stellarator.get_dofs()])
        
        self.res_reference = None
        self.x_reference = None # initially None
        # reference values of objective and gradient
        self.update(self.x)
        self.x_reference = self.x.copy()
        self.res_reference = self.res
        self.dres_reference = self.dres.copy()

        self.Jvals = [np.array([self.res] + self.Jvals_individual)]
        self.dJvals = [ np.array([np.linalg.norm(self.dres, ord=np.inf), np.linalg.norm(self.drescurr, ord=np.inf), np.linalg.norm(self.drescoil, ord=np.inf)]) ]
        
        self.xhistory = [self.x]
        self.Jhistory = [self.res]
        self.dJhistory = [self.dres]

        
        self.output = output
        self.outdir = f"output" + outdir_append + "/"

        if self.iotas_target is None:
            itarget_string = ''
        else:
            itarget_string = ','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_target) for i in rlist])

        if self.iotas_lb is None:
            ilb_string=''
        else:
            ilb_string=','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_lb) for i in rlist])
        if self.iotas_ub is None:
            iub_string = ''
        else:    
            iub_string=','.join('%.16e'%i if i is not None else "-" for i in [i for rlist in MPI.COMM_WORLD.allgather(self.iotas_ub) for i in rlist])

        tftarget_string = ','.join('%.16e'%tf if tf is not None else "-" for tf in [i for rlist in MPI.COMM_WORLD.allgather(self.toroidal_flux_targets) for i in rlist])
        mRtarget_string = ','.join('%.16e'%m if m is not None else "-" for m in [i for rlist in MPI.COMM_WORLD.allgather(self.major_radii_targets) for i in rlist])
        
        if self.initial_areas is None:
            init_areas_string = ''
        else:
            init_areas_string = ','.join('%.16e'%m if m is not None else "-" for m in [i for rlist in MPI.COMM_WORLD.allgather(self.initial_areas) for i in rlist])


        if self.output and self.rank == 0:
            os.makedirs(self.outdir, exist_ok=True)

        MPI.COMM_WORLD.barrier()
        self.out_surface = []
        for i, bs in enumerate(self.boozer_surface_list):
            self.out_surface.append( open(self.outdir + f"out_surface{i}_rank={self.rank}.txt", "w") )
            self.out_surface[i].close()
        for i, bs in enumerate(self.boozer_surface_list):
            self.out_surface[i] = open(self.outdir + f"out_surface{i}_rank={self.rank}.txt", "w")

        if self.output and self.rank==0:

            self.out_x = open(self.outdir + "out_x.txt", "w")
            self.out_x.close()
            self.out_x = open(self.outdir + "out_x.txt", "a")

            self.out_Jvals = open(self.outdir + "out_Jvals.txt", "w")
            self.out_Jvals.close()
            self.out_Jvals = open(self.outdir + "out_Jvals.txt", "a")

            self.out_dJvals = open(self.outdir + "out_dJvals.txt", "w")
            self.out_dJvals.close()
            self.out_dJvals = open(self.outdir + "out_dJvals.txt", "a")



            set_file_logger(self.outdir + "/log.txt")




            out_targets = open(self.outdir + "out_targets.txt", "w")
            out_targets.write("iotas_targets " + itarget_string+"\n")
            out_targets.write("iotas_targets_weight " + str(self.iotas_target_weight) +"\n")
            out_targets.write("tf_targets " + tftarget_string+"\n")
            out_targets.write("tf_weight " + str(self.tf_weight)+"\n")
            out_targets.write("mR_targets " + mRtarget_string+"\n")
            out_targets.write("mr_weight " + str(self.mr_weight)+"\n")
            out_targets.write("initial_areas " + init_areas_string+"\n")
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



    def set_dofs(self, x):
        self.stellarator.set_currents(x[self.idx_curr]*self.current_fak)
        self.stellarator.set_dofs(x[self.idx_coil])
        for bs_tf in self.bs_tf_list: 
            for coil, curr in zip(bs_tf.coils_optim, self.stellarator.currents):
                coil.current.set_value(curr) 
        for bs_boozer_residual in self.bs_boozer_residual_list:
            for coil, curr in zip(bs_boozer_residual.coils_optim, self.stellarator.currents):
                coil.current.set_value(curr) 
        for bs_nonqs in self.bs_nonqs_list:
            for coil, curr in zip(bs_nonqs.coils_optim, self.stellarator.currents):
                coil.current.set_value(curr) 
        for bs_ratio in self.bs_ratio_list:
            for coil, curr in zip(bs_ratio.coils_optim, self.stellarator.currents):
                coil.current.set_value(curr) 
        for boozer_surface in self.boozer_surface_list:
            for coil, curr in zip(boozer_surface.bs.coils_optim, self.stellarator.currents):
                coil.current.set_value(curr) 
    #@profile
    def update(self, x, compute_derivative=True, verbose=True):
        self.x[:] = x
        self.set_dofs(x)
        
        for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
            boozer_surface.surface.set_dofs(reference_surface['dofs'])
            iota0 = reference_surface['iota']
            G0 = reference_surface['G']
            try: 
                if boozer_surface.res['type'] == 'exact':
                    res = boozer_surface.solve_residual_equation_exactly_newton( tol=1e-13, maxiter=30, iota=iota0, G=G0)
                    res['solver'] = 'NEWTON'
                    #if verbose:
                    #    info(f"NEWTON iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e} ")
                else:
                    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                    res['solver'] = 'LS'
                    #if verbose:
                    #    info(f"LS     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
                    if not res['success']:
                        res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=5e-10, maxiter=30, constraint_weight=100., iota=res['iota'], G=res['G'])
                        res['solver'] = 'NEWTON'
                        #if verbose:
                         #    info(f"NEWTON iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
            except:
               boozer_surface.res['success']=False 
            # check self intersecting
            #xs = boozer_surface.surface.cross_section(0., thetas=np.linspace(0,1, 100, endpoint=False))
            #if is_self_intersecting(xs):
            #    info("self intersecting!")
 

#            if not res['success']:
#                info("backtracking: failed surface solve")
#                for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
#                    boozer_surface.surface.set_dofs(reference_surface['dofs'])
#                    iota0 = reference_surface['iota']
#                    G0 = reference_surface['G']
#                
#                if self.res_reference is None:
#                    info("failed to converge on init")
#                    quit()
#
#                self.res = 2*self.res_reference
#                self.dres = -self.dres_reference.copy()
#                info("--------------------------------------------------------------------------------")
#                return



        success = True
        for bs in self.boozer_surface_list:
            success = success and bs.res['success']
        success_list = MPI.COMM_WORLD.allgather(success)

        res_list = [ {'solver':bs.res['solver'], 'type':bs.res['type'], 'success':bs.res['success'], 'iter':bs.res['iter'], 'residual':bs.res['residual'], 'iota':bs.res['iota']} for bs in self.boozer_surface_list ]
        for res, bs in zip(res_list, self.boozer_surface_list):
            if bs.res['type'] != 'exact':
                res['gradient'] = bs.res['gradient']
        res_list = [r for rlist in MPI.COMM_WORLD.allgather(res_list) for r in rlist]
        for res in res_list:
            if verbose: 
                if res['type'] == 'exact':
                    info(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e} ")
                else:
                    info(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
        
        if False in success_list:
            info("backtracking: failed surface solve")
            for reference_surface, boozer_surface in zip(self.boozer_surface_reference, self.boozer_surface_list):
                boozer_surface.surface.set_dofs(reference_surface['dofs'])
                iota0 = reference_surface['iota']
                G0 = reference_surface['G']
            if self.res_reference is None:
                info("failed to converge on init")
                quit()

            self.res = 2*self.res_reference
            self.dres = -self.dres_reference.copy() 
            info("--------------------------------------------------------------------------------")
            return




        xs_list = [bs.surface.cross_section(0., thetas=np.linspace(0, 1, 100, endpoint=False)) for bs in self.boozer_surface_list]
        si_list = [is_self_intersecting(xs) for xs in xs_list]
        all_si_list = [bval for s_list in MPI.COMM_WORLD.allgather(si_list) for bval in s_list]
        if True in all_si_list:
            bool_string = " ".join(['True' if b else 'False' for b in all_si_list])
            info("Self-intersections " + bool_string)

        # I don't need to clear the biotsavart's on the surfaces becasue they were just computed.
        # move to the set_dofs() function
        for J1 in self.J_nonQS:
            J1.clear_cached_properties() 
        for J2 in self.J_iotas:
            J2.clear_cached_properties() 
        for J4 in self.J_major_radii:
            J4.clear_cached_properties() 
        for tf in self.J_toroidal_flux:
            tf.clear_cached_properties() # this also resets the points at which the vector potential is computed
        for J9 in self.J_area:
            J9.clear_cached_properties() # this also resets the points at which the vector potential is computed
        for J11 in self.J_boozer_residual:
            J11.clear_cached_properties()
        for J14 in self.J_nonQS_ratio:
            J14.clear_cached_properties()


        J_nonQS = self.J_nonQS
        J_iotas = self.J_iotas
        J_coil_lengths    = self.J_coil_lengths
        J_major_radii    = self.J_major_radii
        J_distance = self.J_distance
        J_toroidal_flux = self.J_toroidal_flux
        J_arclength = self.J_arclength
        J_curvature = self.J_curvature
        J_msc = self.J_msc
        J_area = self.J_area
        J_nonQS_ratio = self.J_nonQS_ratio
        #self.drescurr = np.zeros(self.idx_curr.shape)
        #self.drescoil = np.zeros(self.idx_coil.shape)

        self.res1 = 0 
        self.dres1curr = np.zeros(self.idx_curr.shape)
        self.dres1coil = np.zeros(self.idx_coil.shape)

        self.res2 = 0
        self.dres2curr = np.zeros(self.idx_curr.shape)
        self.dres2coil = np.zeros(self.idx_coil.shape)

        self.res3 = 0
        self.dres3curr = np.zeros(self.idx_curr.shape)
        self.dres3coil = np.zeros(self.idx_coil.shape)

        self.res4 = 0
        self.dres4curr = np.zeros(self.idx_curr.shape)
        self.dres4coil = np.zeros(self.idx_coil.shape)

        self.res5 = 0
        self.dres5curr = np.zeros(self.idx_curr.shape)
        self.dres5coil = np.zeros(self.idx_coil.shape)

        self.res6 = 0
        self.dres6curr = np.zeros(self.idx_curr.shape)
        self.dres6coil = np.zeros(self.idx_coil.shape)

        self.res7 = 0
        self.dres7curr = np.zeros(self.idx_curr.shape)
        self.dres7coil = np.zeros(self.idx_coil.shape)

        self.res8 = 0
        self.dres8curr = np.zeros(self.idx_curr.shape)
        self.dres8coil = np.zeros(self.idx_coil.shape)

        self.res9 = 0
        self.dres9curr = np.zeros(self.idx_curr.shape)
        self.dres9coil = np.zeros(self.idx_coil.shape)

        self.res10= 0
        self.dres10curr = np.zeros(self.idx_curr.shape)
        self.dres10coil = np.zeros(self.idx_coil.shape)

        self.res11= 0
        self.dres11curr = np.zeros(self.idx_curr.shape)
        self.dres11coil = np.zeros(self.idx_coil.shape)

        self.res12 = 0
        self.dres12curr = np.zeros(self.idx_curr.shape)
        self.dres12coil = np.zeros(self.idx_coil.shape)
        self.diotas_sumcurr = 0
        self.diotas_sumcoil = 0

        self.res13 = 0
        self.dres13curr = np.zeros(self.idx_curr.shape)
        self.dres13coil = np.zeros(self.idx_coil.shape)

        self.res14 = 0
        self.dres14curr = np.zeros(self.idx_curr.shape)
        self.dres14coil = np.zeros(self.idx_coil.shape)

        if self.initial_areas is None: # if no areas provided, compute use the nonQS ratio instead
            self.res14      = np.sum([J14.J()/self.Nsurfaces for J14 in self.J_nonQS_ratio]) 
            if compute_derivative:
                for J14 in J_nonQS_ratio:
                    self.dres14curr += self.stellarator.reduce_current_derivatives(J14.dJ_by_dcoilcurrents()) * self.current_fak/self.Nsurfaces
                    self.dres14coil += self.stellarator.reduce_coefficient_derivatives(J14.dJ_by_dcoefficients())/self.Nsurfaces
        else:
            self.res1      = np.sum([J1.J()/(ar*self.Nsurfaces) for J1, ar in zip(self.J_nonQS, self.initial_areas)]) 
            if compute_derivative:
                for J1, ar in zip(J_nonQS, self.initial_areas):
                    self.dres1curr += self.stellarator.reduce_current_derivatives(J1.dJ_by_dcoilcurrents()) * self.current_fak/(ar*self.Nsurfaces)
                    self.dres1coil += self.stellarator.reduce_coefficient_derivatives(J1.dJ_by_dcoefficients())/(ar*self.Nsurfaces)
 

        if self.iotas_target is not None:
            self.res2      = self.iotas_target_weight * np.sum([0.5 * (J2.J()-iotas_target)**2 if iotas_target is not None else 0. for J2, iotas_target in zip(self.J_iotas, self.iotas_target)])
            if compute_derivative:
                for J2, iotas_target in zip(J_iotas, self.iotas_target):
                    if iotas_target is not None:
                        self.dres2curr += self.iotas_target_weight *     self.stellarator.reduce_current_derivatives([ (J2.J() - iotas_target) * dj_dc for dj_dc in J2.dJ_by_dcoilcurrents()]) * self.current_fak
                        self.dres2coil += self.iotas_target_weight * self.stellarator.reduce_coefficient_derivatives([ (J2.J() - iotas_target) * dj_dc for dj_dc in J2.dJ_by_dcoefficients()])
       
        self.res3 = self.lengthbound_weight * self.J_lengthbound_penalty.J()
        if compute_derivative:
            self.dres3coil += self.lengthbound_weight * self.stellarator.reduce_coefficient_derivatives( self.J_lengthbound_penalty.dJ() )

        self.res4      = 0.5 * self.mr_weight * sum(  (J3.J() - l)**2 if l is not None else 0. for (J3, l) in zip(J_major_radii, self.major_radii_targets))
        if compute_derivative:
            for J4, l in zip(J_major_radii, self.major_radii_targets):
                if l is not None:
                    self.dres4curr += self.mr_weight *     self.stellarator.reduce_current_derivatives( [ (J4.J()-l) * dj_dc for dj_dc in J4.dJ_by_dcoilcurrents()] ) * self.current_fak
                    self.dres4coil += self.mr_weight * self.stellarator.reduce_coefficient_derivatives( [ (J4.J()-l) * dj_dc for dj_dc in J4.dJ_by_dcoefficients()] ) 
        
        if self.distance_weight > 1e-15:
            self.res5 = self.distance_weight * J_distance.J()
            if compute_derivative:
                self.dres5coil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(J_distance.dJ())
        else:
            self.res5 = 0.
        
        self.res6      = 0.5 * self.tf_weight  * sum(  (J6.J() - l)**2 if l is not None else 0. for (J6, l) in zip(J_toroidal_flux, self.toroidal_flux_targets))
        if compute_derivative:
            for J6, l in zip(J_toroidal_flux, self.toroidal_flux_targets):
                if l is not None:
                    self.dres6curr += self.tf_weight *     self.stellarator.reduce_current_derivatives( [ (J6.J()-l) * dj_dc for dj_dc in J6.dJ_by_dcoilcurrents()] ) * self.current_fak
                    self.dres6coil += self.tf_weight * self.stellarator.reduce_coefficient_derivatives( [ (J6.J()-l) * dj_dc for dj_dc in J6.dJ_by_dcoefficients()] ) 

        if self.arclength_weight > 1e-15:
            self.res7 = self.arclength_weight * sum([J7.J() for J7 in J_arclength])
            if compute_derivative:
                self.dres7coil += self.arclength_weight * self.stellarator.reduce_coefficient_derivatives([J7.dJ_by_dcoefficients() for J7 in J_arclength])
        
        if self.curvature_weight > 1e-15:
            self.res8 = self.curvature_weight * sum([J8.J() for J8 in J_curvature])
            if compute_derivative:
                self.dres8coil += self.curvature_weight * self.stellarator.reduce_coefficient_derivatives([J8.dJ() for J8 in J_curvature])
        
        if self.iotas_lb is not None:
            self.res9      = self.iotas_bound_weight * np.sum([0.5 * max([iotas_lb-J2.J(),0])**2 if iotas_lb is not None else 0. for J2, iotas_lb in zip(self.J_iotas, self.iotas_lb)])
            if compute_derivative:
                for J2, iotas_lb in zip(J_iotas, self.iotas_lb):
                    if iotas_lb is not None:
                        self.dres9curr += self.iotas_bound_weight *     self.stellarator.reduce_current_derivatives([ -max([iotas_lb - J2.J() ,0]) * dj_dc for dj_dc in J2.dJ_by_dcoilcurrents()]) * self.current_fak
                        self.dres9coil += self.iotas_bound_weight * self.stellarator.reduce_coefficient_derivatives([ -max([iotas_lb - J2.J() ,0]) * dj_dc for dj_dc in J2.dJ_by_dcoefficients()])
        
        if self.iotas_ub is not None:
            self.res10      = self.iotas_bound_weight * np.sum([0.5 * max([J2.J()-iotas_ub,0])**2 if iotas_ub is not None else 0. for J2, iotas_ub in zip(self.J_iotas, self.iotas_ub)])
            if compute_derivative:
                for J2, iotas_ub in zip(J_iotas, self.iotas_ub):
                    if iotas_ub is not None:
                        self.dres10curr += self.iotas_bound_weight *     self.stellarator.reduce_current_derivatives([ max([J2.J()-iotas_ub ,0]) * dj_dc for dj_dc in J2.dJ_by_dcoilcurrents()]) * self.current_fak
                        self.dres10coil += self.iotas_bound_weight * self.stellarator.reduce_coefficient_derivatives([ max([J2.J()-iotas_ub ,0]) * dj_dc for dj_dc in J2.dJ_by_dcoefficients()])

        if self.residual_weight is not None:
            self.res11    = np.sum( [res_weight*res.J() if res_weight is not None else 0. for res_weight, res in zip(self.residual_weight, self.J_boozer_residual)] )
            if compute_derivative:
                for res_weight, bres in zip(self.residual_weight, self.J_boozer_residual):
                    if res_weight is not None:
                        self.dres11curr += res_weight *     self.stellarator.reduce_current_derivatives(bres.dJ_by_dcoilcurrents()) * self.current_fak
                        self.dres11coil += res_weight * self.stellarator.reduce_coefficient_derivatives(bres.dJ_by_dcoefficients())

        if self.iotas_avg_target is not None:
            self.iotas_sum = np.sum([J2.J() for J2 in self.J_iotas])
            if compute_derivative:
                for J2 in J_iotas:
                    self.diotas_sumcurr +=     self.stellarator.reduce_current_derivatives([ dj_dc for dj_dc in J2.dJ_by_dcoilcurrents()]) * self.current_fak
                    self.diotas_sumcoil += self.stellarator.reduce_coefficient_derivatives([ dj_dc for dj_dc in J2.dJ_by_dcoefficients()])

        if self.msc_weight > 1e-15:
            self.res13 = self.msc_weight * sum([J13.J() for J13 in J_msc])
            if compute_derivative:
                self.dres13coil += self.msc_weight * self.stellarator.reduce_coefficient_derivatives([J13.dJ() for J13 in J_msc])



        self.Jvals_individual  = [self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res10, self.res11, self.res13, self.res14]
        self.dJvals_individualcurr = [self.dres1curr, self.dres2curr, self.dres3curr, self.dres4curr, self.dres5curr, self.dres6curr, self.dres7curr, self.dres8curr, self.dres9curr, self.dres10curr, self.dres11curr, self.dres13curr, self.dres14curr]
        self.dJvals_individualcoil = [self.dres1coil, self.dres2coil, self.dres3coil, self.dres4coil, self.dres5coil, self.dres6coil, self.dres7coil, self.dres8coil, self.dres9coil, self.dres10coil, self.dres11coil, self.dres13coil, self.dres14coil]

        # communicate surface related terms in objective
        values = MPI.COMM_WORLD.allgather([self.Jvals_individual, self.dJvals_individualcurr, self.dJvals_individualcoil])


        idx_coil_terms    = [2, 4, 6, 7, 11]
        idx_surface_terms = [0, 1, 3, 5, 8, 9, 10, 12]

        self.res = sum([self.Jvals_individual[idx] for idx in idx_coil_terms]) 
        self.drescurr = sum([self.dJvals_individualcurr[idx] for idx in idx_coil_terms])
        self.drescoil = sum([self.dJvals_individualcoil[idx] for idx in idx_coil_terms])

        for v in values:
            self.res+=sum([v[0][idx] for idx in idx_surface_terms])
            self.drescurr+=sum([v[1][idx] for idx in idx_surface_terms])
            self.drescoil+=sum([v[2][idx] for idx in idx_surface_terms])


        # add on average iotas term
        if self.iotas_avg_target is not None:
            all_iotas = MPI.COMM_WORLD.allgather([ self.iotas_sum, self.diotas_sumcurr, self.diotas_sumcoil, len(self.boozer_surface_list)])
            self.iotas_avg= sum([r[0] for r in all_iotas])/self.Nsurfaces
            self.diotas_avg_curr = sum([r[1] for r in all_iotas])/self.Nsurfaces
            self.diotas_avg_coil = sum([r[2] for r in all_iotas])/self.Nsurfaces
            self.res12     = 0.5 * self.iotas_avg_weight * ( self.iotas_avg - self.iotas_avg_target)**2
            self.res12curr =       self.iotas_avg_weight * ( self.iotas_avg - self.iotas_avg_target) * self.diotas_avg_curr
            self.res12coil =       self.iotas_avg_weight * ( self.iotas_avg - self.iotas_avg_target) * self.diotas_avg_coil
            
            self.res+=self.res12
            self.drescurr+=self.res12curr
            self.drescoil+=self.res12coil



        self.dres = np.concatenate([self.drescurr, self.drescoil])

        # update all res values to communicated ones
        all_res = [v[0] for v in values]
        self.res1 =sum([res[0] for res in all_res]) 
        self.res2 =sum([res[1] for res in all_res]) 
        #self.res3 =sum([res[2] for res in all_res]) 
        self.res4 =sum([res[3] for res in all_res]) 
        #self.res5 =sum([res[4] for res in all_res]) 
        self.res6 =sum([res[5] for res in all_res]) 
        #self.res7 =sum([res[6] for res in all_res]) 
        #self.res8 =sum([res[7] for res in all_res]) 
        self.res9 =sum([res[8] for res in all_res]) 
        self.res10=sum([res[9] for res in all_res]) 
        self.res11=sum([res[10] for res in all_res])
        #self.res12 =sum([res[11] for res in all_res]) 
        self.res14=sum([res[12] for res in all_res])
        self.Jvals_individual  = [self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res10, self.res11, self.res12, self.res13, self.res14]

        if verbose:
            info("--------------------------------------------------------------------------------")

    
    def min_dist(self):
        res = 1e10
        for i in range(len(self.stellarator.coils)):
            gamma1 = self.stellarator.coils[i].gamma()
            for j in range(i):
                gamma2 = self.stellarator.coils[j].gamma()
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res = min(res, np.min(dists))
        return res




    def callback(self, x, verbose=True):
        # sanity check that the line search settled on the last state
        assert np.allclose(self.x, x, rtol=0, atol=0)
        
        # check self.x is the same at all ranks
        all_x = MPI.COMM_WORLD.allgather(self.x)
        for xx in all_x:
            assert np.allclose(self.x, xx, rtol=0, atol=0)
        
        

        # sanity check that all the surfaces have converged
        for booz_s in self.boozer_surface_list:
            assert booz_s.res['success']

        self.iter += 1
        
        self.Jvals.append(np.array([self.res] + self.Jvals_individual))
        self.dJvals.append(np.array([np.linalg.norm(self.dres, ord=np.inf), np.linalg.norm(self.drescurr, ord=np.inf), np.linalg.norm(self.drescoil, ord=np.inf)]))
        
        self.xhistory.append(x)
        self.Jhistory.append(self.res)
        self.dJhistory.append(self.dres)

        # update reference dofs used as initial guesses during line search.
        self.boozer_surface_reference = [{"dofs": boozer_surface.surface.get_dofs(),
                                         "iota": boozer_surface.res["iota"],
                                          "G": boozer_surface.res["G"]} for boozer_surface in self.boozer_surface_list]
        # update reference res, and dres
        self.x_reference[:] = x[:]
        self.res_reference = self.res
        self.dres_reference[:] = self.dres[:]

        iotas = [ res['iota'] for res in self.boozer_surface_reference ]
        volumes = [Volume(bs.surface, self.stellarator).J() for bs in self.boozer_surface_list]
        areas = [Area(bs.surface, self.stellarator).J() for bs in self.boozer_surface_list]
        aspect_ratios = [Aspect_ratio(bs.surface, self.stellarator).J() for bs in self.boozer_surface_list]

        mR = [ R.J() for R in self.J_major_radii ]
        mr = [ np.sqrt(np.abs(vol)/(2*R*np.pi**2)) for R, vol in zip(mR, volumes) ]
        
        sa_list = [ np.min( np.linalg.norm( bs.surface.normal(), axis=-1 ) ) for bs in self.boozer_surface_list ]
        tf_list = [ tf.J() for tf in self.J_toroidal_flux]
        
        # communicate data before creating strings
        iotas = [i for d in MPI.COMM_WORLD.allgather(iotas) for i in d]
        mr = [i for d in MPI.COMM_WORLD.allgather(mr) for i in d]
        mR = [i for d in MPI.COMM_WORLD.allgather(mR) for i in d]
        volumes = [i for d in MPI.COMM_WORLD.allgather(volumes) for i in d]
        areas = [i for d in MPI.COMM_WORLD.allgather(areas) for i in d]
        aspect_ratios = [i for d in MPI.COMM_WORLD.allgather(aspect_ratios) for i in d]
        sa_list = [i for d in MPI.COMM_WORLD.allgather(sa_list) for i in d]
        tf_list = [i for d in MPI.COMM_WORLD.allgather(tf_list) for i in d]


        i_string = ' '.join('%.8e'%i for i in iotas )
        mr_string = ' '.join('%.8e'%m for m in mr )
        mR_string = ' '.join('%.8e'%m for m in mR )
        vol_string = ' '.join('%.8e'%vol for vol in volumes )
        area_string = ' '.join('%.8e'%ar for ar in areas)
        ar_string = ' '.join('%.8e'%ar for ar in aspect_ratios)
        sa_string = ' '.join('%.8e'%sa for sa in sa_list)
        tf_string = ' '.join('%.8e'%tf for tf in tf_list)
        

        if self.iotas_target is None:
            itarget_string = None
        else:
            itarget_string  = ' '.join('%.8e'%i if i is not None   else "--------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.iotas_target) for i in d] )
        
        if self.iotas_lb is None:
            ilb_string = None
        else:
            ilb_string      = ' '.join('%.8e'%i if i is not None  else "--------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.iotas_lb) for i in d] )
        
        if self.iotas_ub is None:
            iub_string = None
        else:
            iub_string      = ' '.join('%.8e'%i if i is not None  else "--------------" for i in [i for d in MPI.COMM_WORLD.allgather(self.iotas_ub) for i in d])
        mRtarget_string = ' '.join('%.8e'%m if m is not None   else "--------------" for m in [i for d in MPI.COMM_WORLD.allgather(self.major_radii_targets ) for i in d] )
        tftarget_string = ' '.join('%.8e'%tf if tf is not None else "--------------" for tf in [i for d in MPI.COMM_WORLD.allgather(self.toroidal_flux_targets )  for i in d])


        def compute_non_quasisymmetry_L2(in_surface):
            from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

            bs = BiotSavart(self.stellarator.coils, self.stellarator.currents)
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
        ratio = [i for d in MPI.COMM_WORLD.allgather(ratio) for i in d]

        ratio_string=' '.join('%.8e'%r for r in ratio )

        maxkappa = max([np.max(np.abs(coil.kappa())) for coil in self.stellarator._base_coils])
        minarclength = min([np.min(np.abs(coil.incremental_arclength())) for coil in self.stellarator._base_coils]) 
        length = sum([J.J() for J in self.J_coil_lengths]) 
        msc = ' '.join([f"{MeanSquareCurvature(coil, self.msc_max).msc():.8e}" for coil in self.stellarator._base_coils])
        cv = ' '.join([f"{np.max(np.abs(coil.kappa())):.8e}" for coil in self.stellarator._base_coils])

        res_list = [ {'type':bs.res['type'], 'success':bs.res['success'], 'iter':bs.res['iter'], 'residual':bs.res['residual']} for bs in self.boozer_surface_list ]
        for res, bs in zip(res_list, self.boozer_surface_list):
            if bs.res['type'] != 'exact':
                res['gradient'] = bs.res['gradient']
        res_list = [r for rlist in MPI.COMM_WORLD.allgather(res_list) for r in rlist]
        
        info("################################################################################")
        info(f"Iteration {self.iter}")
        info(f"Objective value:             {self.res:.8e}")
        info("")
        info( "                              nonQS           iotas           coil lengths    major radius    minimum distance       tf") 
        info(f"Objective values1:            {self.res14 if self.initial_areas is None else self.res1:.8e}, {self.res2:.8e}, {self.res3:.8e}, {self.res4:.8e}, {self.res5:.8e}, {self.res6:.8e}")
        info("")
        info( "                              arclength       curvature        iota_lb         iota_ub       boozerLS       iota_avg       mean square curvature") 
        info(f"Objective values2:            {self.res7:.8e}, {self.res8:.8e}, {self.res9:.8e}, {self.res10:.8e}, {self.res11:.8e}, {self.res12:.8e}, {self.res13:.8e}")
        info("")
        info(f"Gradient values:             {self.dJvals[-1][0]:.6e}, {self.dJvals[-1][1]:.6e}, {self.dJvals[-1][2]:.6e}")
        info(f"non_qs/qs:                   {ratio_string}") 
        info(f"iotas:                       {i_string}") 
        info(f"iotas_avg:                   {np.mean(iotas):.8e}") 
        info(f"iotas_avg_target:            {self.iotas_avg_target:.8e}") 
        if self.iotas_target is not None:
            info(f"iotas_target:                {itarget_string}") 
        if self.iotas_lb is not None:
            info(f"iotas_lb:                    {ilb_string}") 
        if self.iotas_ub is not None:
            info(f"iotas_ub:                    {iub_string}") 
        info(f"major radii:                 {mR_string}") 
        info(f"major radii target:          {mRtarget_string}") 
        info(f"minor radii:                 {mr_string}") 
        info(f"toroidal flux:               {tf_string}") 
        info(f"toroidal flux targets:       {tftarget_string}") 
        info(f"aspect ratios:               {ar_string}") 
        info(f"minimum surface area el:     {sa_string}") 
        info(f"volume:                      {vol_string}") 
        info(f"area:                        {area_string}") 
        info(f"minimum distance: {self.min_dist():.6e}")
        info(f"maximum curvature: {maxkappa:.6e}")
        info(f"curvature: {cv}")
        info(f"mean square curvature: {msc}")
        info(f"minimum arclength: {minarclength:.6e}")
        info(f"total coil length: {length:.6e}")

        for res in res_list:
            if res['type'] == 'exact':
                info(f"iter={res['iter']}, success={res['success']}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
            else:
                info(f"iter={res['iter']}, success={res['success']}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
 
        info("################################################################################")

        for i, bs in enumerate(self.boozer_surface_list):
           np.savetxt( self.out_surface[i], np.concatenate( (bs.surface.get_dofs(), [bs.res['iota'], bs.res['G']]) ).reshape((1,-1)))

        if self.output and self.rank==0:
            np.savetxt(self.out_x, self.x.reshape((1,-1)))
            np.savetxt(self.out_Jvals, self.Jvals[-1].reshape((1,-1)))
            np.savetxt(self.out_dJvals, self.dJvals[-1].reshape((1,-1)))
        if self.iter % 25 == 0 :
            self.plot('iteration-%04i.png' % self.iter)
    
    def plot(self, filename=None):
        if filename is None:
            raise Exception("Need to provide filename for plotting")
        from simsopt.geo.curve import curves_to_vtk
        for idx, bs in enumerate(self.boozer_surface_list):
            bs.surface.to_vtk(self.outdir + filename + f"surface_{idx}_rank={self.rank}")

        if self.rank == 0:
            curves_to_vtk(self.stellarator.coils, self.outdir + filename)
        
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            for i in range(0, len(self.stellarator.coils)):
                ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])
            ax.view_init(elev=90., azim=0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            for bs in self.boozer_surface_list:
                gamma = bs.surface.gamma()
                X = np.concatenate((gamma[:,:,0], gamma[:,0,0].reshape((-1,1))), axis=1 )
                Y = np.concatenate((gamma[:,:,1], gamma[:,0,1].reshape((-1,1))), axis=1 )
                Z = np.concatenate((gamma[:,:,2], gamma[:,0,2].reshape((-1,1))), axis=1 )
                ax.plot_surface(X, Y, Z, linewidth=1, edgecolors='k')

            ax = fig.add_subplot(1, 2, 2, projection="3d")
            for i in range(0, len(self.stellarator.coils)):
                ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])

            for bs in self.boozer_surface_list:
                gamma = bs.surface.gamma()
                X = np.concatenate((gamma[:,:,0], gamma[:,0,0].reshape((-1,1))), axis=1 )
                Y = np.concatenate((gamma[:,:,1], gamma[:,0,1].reshape((-1,1))), axis=1 )
                Z = np.concatenate((gamma[:,:,2], gamma[:,0,2].reshape((-1,1))), axis=1 )
                ax.plot_surface(X, Y, Z, linewidth=1, edgecolors='k')

            ax.view_init(elev=0., azim=45)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            plt.savefig(self.outdir + filename, dpi=300)
            plt.close()

            if "DISPLAY" in os.environ:
                try:
                    import mayavi.mlab as mlab
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "\n\nPlease install mayavi first. On a mac simply do \n" +
                        "   pip3 install mayavi PyQT5\n" +
                        "On Ubuntu run \n" +
                        "   pip3 install mayavi\n" +
                        "   sudo apt install python3-pyqt4\n\n"
                    )

                mlab.options.offscreen = True
                colors = [
                    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
                    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
                    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
                    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
                    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
                    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
                    (0.8, 0.7254901960784313, 0.4549019607843137),
                    (0.39215686274509803, 0.7098039215686275, 0.803921568627451)
                ]

                mlab.figure(bgcolor=(1, 1, 1))
                for i in range(0, len(self.stellarator.coils)):
                    gamma = self.stellarator.coils[i].gamma()
                    gamma = np.concatenate((gamma, [gamma[0,:]]))
                    mlab.plot3d(gamma[:, 0], gamma[:, 1], gamma[:, 2], color=colors[i%len(self.stellarator._base_coils)])

                for idx, boozer_surface in enumerate(self.boozer_surface_list):
                    # revert to reference states - zeroth order continuation
                    s = boozer_surface.surface
                    gamma = s.gamma()
                    Bn = np.linalg.norm(boozer_surface.bs.B().reshape((gamma.shape[0], gamma.shape[1], -1)), axis=-1)
                    Bn = np.concatenate((Bn, Bn[:,0].reshape((-1,1))), axis=1)
                    mlab.mesh(np.concatenate((gamma[:,:,0], gamma[:,0,0].reshape((-1,1)) ), axis = 1), 
                              np.concatenate((gamma[:,:,1], gamma[:,0,1].reshape((-1,1)) ), axis = 1),
                              np.concatenate((gamma[:,:,2], gamma[:,0,2].reshape((-1,1)) ), axis = 1), scalars=Bn)

                    mlab.mesh(np.concatenate((gamma[:,:,0], gamma[:,0,0].reshape((-1,1)) ), axis = 1), 
                              np.concatenate((gamma[:,:,1], gamma[:,0,1].reshape((-1,1)) ), axis = 1),
                              np.concatenate((gamma[:,:,2], gamma[:,0,2].reshape((-1,1)) ), axis = 1),  representation='wireframe', color=(0, 0, 0))
                
                count = 0
                for i in [-90, -45, 0, 45, 90]:
                    for j in [-90, -45, 0, 45, 90]:
                        mlab.view(azimuth=i, elevation=j)
                        mlab.savefig(self.outdir + f"mayavi_{count}_" + filename, magnification=4)
                        count += 1
                mlab.close()
