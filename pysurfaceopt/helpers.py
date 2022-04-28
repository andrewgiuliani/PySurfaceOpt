import numpy as np
import os
from simsopt.util.zoo import get_ncsx_data
from simsopt.field.coil import ScaledCurrent, Current, coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.boozersurface import BoozerSurface, boozer_surface_residual
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from pysurfaceopt.surfaceobjectives import ToroidalFlux, MajorRadius, Volume
from ground.base import get_context
from bentley_ottmann.planar import contour_self_intersects


def get_stageII_data(Nt_coils=16, Nt_ma=10, ppp=20, length=18):
    order = Nt_coils
    
    assert length == 18 or length == 20 or length == 22 or length == 24
    
    if length == 18:
        ig = 7
    elif length == 20:
        ig = 6
    elif length == 22:
        ig = 4
    elif length == 24:
        ig = 2
    else:
        quit()

    DIR = os.path.dirname(os.path.realpath(__file__))
    c0 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_0.txt')
    c1 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_1.txt')
    c2 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_2.txt')
    c3 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_3.txt')
    currents = [ c0, c1, c2, c3 ]
    base_currents = [Current(c*1e5) for c in currents]

    dofs0 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_0.txt')
    dofs1 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_1.txt')
    dofs2 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_2.txt')
    dofs3 = np.loadtxt(DIR + f'/data_stageII/coil_data/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_3.txt')

    base_curves = [CurveXYZFourier(order*ppp, order) for i in range(4)]
    base_curves[0].x = dofs0
    base_curves[1].x = dofs1
    base_curves[2].x = dofs2
    base_curves[3].x = dofs3

    return (base_curves, base_currents)


def get_stageIII_data(coilset='nine', length=18):
    order=16
    ppp=10

    assert length == 18 or length == 20 or length == 22 or length == 24
    
    DIR = os.path.dirname(os.path.realpath(__file__))
    dofs = np.loadtxt(DIR + f'/data_stageIII/{coilset}/len{length}.txt')
    currents = dofs[:4].tolist()
    geo_dofs = dofs[4:].reshape((4, -1))

    base_currents = [ScaledCurrent(Current(c), 1/(4*np.pi*1e-7)) for c in currents]
    base_curves = [CurveXYZFourier(order*ppp, order) for i in range(4)]
    for i in range(4):
        base_curves[i].x = geo_dofs[i, :]

    return (base_curves, base_currents)

def get_stageIII_problem(coilset='nine', length=18, verbose=False, output=False):
    ts_dict={18:'1639707710.6463501', 20:'1639796640.5101252', 22:'1642522947.7884622', 24: '1642523475.5701194'}
    mr = np.sqrt(0.5678672050465505/ ( 2 * np.pi**2 ) )
    vol_list = -2. * np.pi**2 * np.linspace(0.01,mr, 9 if coilset=='nine' else 5) ** 2
    
    base_curves, base_currents = get_stageIII_data(coilset, length)

    nfp = 2
    stellsym = True
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    
    # load initial guesses for stageIII surfaces
    boozer_surface_list=[]
    tol=1e-13
    ntor=10
    mpol=10
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
    thetas = np.linspace(0, 1,   2*mpol+1, endpoint=False)
    nquadphi = phis.size
    nquadtheta=thetas.size
    exact=True
    Nt_coils=16
    time_stamp = ts_dict[length]
    DIR = os.path.dirname(os.path.realpath(__file__))
    with open(DIR + "/data_stageII" + f"/surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data_stageII"        + f"/iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))


    for idx in range(9 if coilset=='nine' else 5):
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
        
        iota0 = iotaG[idx,0]
        G0 = iotaG[idx,1]
        
        bs = BiotSavart(coils)
        ll = Volume(s)
        
        # need to actually store target surface label for BoozerLS surfaces
        target = vol_list[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
        r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
        if verbose:
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={ll.J():.3f}, |label error|={np.abs(ll.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
        boozer_surface_list.append(boozer_surface)


    for idx, booz_surf in enumerate(boozer_surface_list):
        booz_surf.bs = BiotSavart(coils)
        booz_surf.targetlabel = vol_list[idx]
    
    ############################################################################
    ## SET THE TARGET IOTAS, MAJOR RADIUS, TOROIDAL FLUX                      ##
    ############################################################################
    
    mr_target_dict = {18: 1.0044683763529081e+00, 20: 1.0044901967769049e+00, 22: 1.0044959075472701e+00, 24: 1.0044957607222651e+00}
    tf_target_dict = {18:-8.6969394054304812e-05, 20:-9.2487854903688596e-05, 22:-9.1817899465679867e-05, 24:-8.9795375467393711e-05}
    
    mr_target   = [None for bs in boozer_surface_list]
    tf_target   = [None for bs in boozer_surface_list]

    mr_target[0] = mr_target_dict[length] 
    tf_target[0] = tf_target_dict[length]
    
    ############################################################################
    ## SET THE INITIAL WEIGHTS, TARGET CURVATURE AND TARGET TOTAL COIL LENGTH ##
    ############################################################################
    kappa_weight_dict =       {18: 1e-5, 20:1e-5, 22:1e-5, 24:1e-5}
    msc_weight_dict =         {18: 1e-5, 20:1e-5, 22:1e-5, 24:1e-5}
    lengthbound_weight_dict = {18:1e-4, 20:1e-4, 22:1e-4, 24:1e-4}
    min_dist_weight_dict =    {18:1e-2, 20:1e-1, 22:1e-2, 24:1e-2}
    alen_weight_dict =        {18:1e-7, 20:1e-7, 22:1e-8, 24:1e-8}
    tf_weight_dict =          {18:1e6, 20:1e6, 22:1e6, 24:1e6}
    
    KAPPA_MAX = 5.
    KAPPA_WEIGHT = kappa_weight_dict[length]
    
    MSC_MAX = 5.
    MSC_WEIGHT = msc_weight_dict[length] 
    
    LENGTHBOUND = length
    LENGTHBOUND_WEIGHT = lengthbound_weight_dict[length] 
    
    MIN_DIST = 0.1
    MIN_DIST_WEIGHT = min_dist_weight_dict[length] 
    
    ALEN_WEIGHT = alen_weight_dict[length] 
    
    IOTAS_AVG_TARGET = -0.42
    IOTAS_AVG_WEIGHT = 1.
    TF_WEIGHT = tf_weight_dict[length] 
    MR_WEIGHT = 1.
    
    from pysurfaceopt.problem import SurfaceProblem
    problem = SurfaceProblem(boozer_surface_list, base_curves, base_currents, coils,
                             major_radii_targets=mr_target, toroidal_flux_targets=tf_target, 
                             iotas_avg_weight=IOTAS_AVG_WEIGHT, iotas_avg_target=IOTAS_AVG_TARGET, mr_weight=MR_WEIGHT, tf_weight=TF_WEIGHT,
                             minimum_distance=MIN_DIST, kappa_max=KAPPA_MAX, lengthbound_threshold=LENGTHBOUND,
                             msc_max=MSC_MAX, msc_weight=MSC_WEIGHT,
                             distance_weight=MIN_DIST_WEIGHT, curvature_weight=KAPPA_WEIGHT, lengthbound_weight=LENGTHBOUND_WEIGHT, arclength_weight=ALEN_WEIGHT,
                             outdir_append="", output=output)
    return problem


# default tol for BoozerLS is 1e-10
# default tol for BoozerExact is 1e-13
def compute_surfaces_in_NCSX(mpol=10, ntor=10, exact=True, Nt_coils=12, write_to_file=False, vol_list=None, tol=1e-13):
    PPP = 20
    base_curves, base_currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    base_currents = [Current(curr.x*4 * np.pi * 1e-7) for curr in base_currents]
    base_currents = [ScaledCurrent(curr, 1/(4 * np.pi * 1e-7)) for curr in base_currents]
 
    nfp = ma.nfp
    stellsym = True
    coils = coils_via_symmetries(base_curves, base_currents, ma.nfp, stellsym)
    
    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), nquadphi, endpoint=False)
        thetas = np.linspace(0, 1, nquadtheta, endpoint=False)
    
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.10, flip_theta=True)
    iota0 = -0.4
    
    if vol_list is None:
        vol_list = np.linspace(-0.162, -3.3, 10)
    curr_sum = np.sum([curr.x for curr in base_currents])
    G0 = 2. * np.pi * 2* nfp * curr_sum / (2 * np.pi)
    boozer_surface_list = []
    boozer_surface_dict = []
    
    backup_dofs = s.get_dofs()
    backup_iota = iota0
    backup_G = G0
        
    for idx,target in enumerate(vol_list):
        bs = BiotSavart(coils)
        label = Volume(s)
        boozer_surface = BoozerSurface(bs, s, label, target)
        try:
            s.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            if exact:
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")

            if res['success']:
                boozer_surface_list.append(boozer_surface)
                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G'], 'label': boozer_surface.surface.volume(), 'target': target})
                backup_dofs = s.get_dofs().copy()
                backup_iota = res['iota']
                backup_G = res['G']
                s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        except Exception as inst:
            print("Surface solver exception: ", type(inst))

    if write_to_file:
        DIR = os.path.dirname(os.path.realpath(__file__))
        import time
        ts = time.time()
        with open(DIR + "/data_ncsx/" + f"surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
        with open(DIR + '/data_ncsx/' +        f"iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s\n' % (surf_dict['iota']))
                f.write('%s\n' % (surf_dict['G']))
        with open(DIR + '/data_ncsx/' +        f"voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s %s\n' % (surf_dict['target'], surf_dict['label']))

    return boozer_surface_list, base_curves, base_currents, coils

def load_surfaces_in_NCSX(mpol=10, ntor=10, stellsym=True, Nt_coils=6, idx_surfaces=np.arange(10), exact=True, time_stamp=None, tol=1e-13):
    nfp = 3    
    if exact:
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1,   2*mpol+1, endpoint=False)
    else:
        phis = np.linspace(0, 1/(2*nfp), ntor+1+5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1+5, endpoint=False)

    nquadphi   = phis.size
    nquadtheta = thetas.size

    PPP = 20
    DIR = os.path.dirname(os.path.realpath(__file__))
    with open(DIR + "/data_ncsx" + f"/surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data_ncsx"        + f"/iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))
    with open(DIR + '/data_ncsx' + f"/voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt") as f:
        inlabels = np.loadtxt(f)
        vol_targets = inlabels[:,0]

    base_curves, base_currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    base_currents = [Current(curr.x*4 * np.pi * 1e-7) for curr in base_currents]
    base_currents = [ScaledCurrent(curr, 1/(4 * np.pi * 1e-7)) for curr in base_currents]
    
    nfp = ma.nfp
    stellsym = True
    coils = coils_via_symmetries(base_curves, base_currents, ma.nfp, stellsym)

    boozer_surface_list = []
    for idx in idx_surfaces:

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
        
        iota0 = iotaG[idx,0]
        G0 = iotaG[idx,1]
        
        bs = BiotSavart(coils)
        ll = Volume(s)

        # need to actually store target surface label for BoozerLS surfaces
        target = vol_targets[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        if exact:
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
            r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={ll.J():.3f}, |label error|={np.abs(ll.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
        else:
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', hessian=True)
            r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={ll.J():.3f}, |label error|={np.abs(ll.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
        boozer_surface_list.append(boozer_surface)
    return boozer_surface_list, base_curves, base_currents, coils


def compute_surfaces_in_stageII(mpol=10, ntor=10, exact=True, Nt_coils=16, write_to_file=False, vol_list=None, tol=1e-13, length=18):
    nfp = 2
    stellsym = True


    PPP = 10
    _, _, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    base_curves, base_currents = get_stageII_data(length=length, Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    base_currents = [Current(curr.x*4 * np.pi * 1e-7) for curr in base_currents]
    base_currents = [ScaledCurrent(curr, 1/(4 * np.pi * 1e-7)) for curr in base_currents]
 
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    
    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), nquadphi, endpoint=False)
        thetas = np.linspace(0, 1, nquadtheta, endpoint=False)
    
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.10, flip_theta=True)
    iota0 = -0.4
    
    if vol_list is None:
        vol_list = np.linspace(-0.162, -3.3, 10)
    curr_sum = np.sum([curr.x for curr in base_currents])
    G0 = 2. * np.pi * curr_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
    boozer_surface_list = []
    boozer_surface_dict = []

    backup_dofs = s.get_dofs()
    backup_iota = iota0
    backup_G = G0
    
    import ipdb;ipdb.set_trace()
    for idx,target in enumerate(vol_list):
        bs = BiotSavart(coils)
        label = Volume(s)
        boozer_surface = BoozerSurface(bs, s, label, target)
        try:
            s.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            if exact:
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")

            if res['success']:
                boozer_surface_list.append(boozer_surface)
                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G'], 'label': boozer_surface.surface.volume(), 'target': target})
                backup_dofs = s.get_dofs().copy()
                backup_iota = res['iota']
                backup_G = res['G']
                s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        except Exception as inst:
            print("Surface solver exception: ", type(inst))

    if write_to_file:
        DIR = os.path.dirname(os.path.realpath(__file__))
        import time
        ts = time.time()
        with open(DIR + "/data_stageII/" + f"surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
        with open(DIR + '/data_stageII/' +        f"iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s\n' % (surf_dict['iota']))
                f.write('%s\n' % (surf_dict['G']))
        with open(DIR + '/data_stageII/' +        f"voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s %s\n' % (surf_dict['target'], surf_dict['label']))

    return boozer_surface_list, base_curves, base_currents, coils

def load_surfaces_in_stageII(length=18, mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=np.arange(10), exact=True, time_stamp=None, tol=1e-13, verbose=False):
    PPP = 10
    nfp = 2
    stellsym = True
    if exact:
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1,   2*mpol+1, endpoint=False)
    else:
        phis = np.linspace(0, 1/(2*nfp), ntor+1+5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1+5, endpoint=False)

    nquadphi   = phis.size
    nquadtheta = thetas.size

    DIR = os.path.dirname(os.path.realpath(__file__))
    with open(DIR + "/data_stageII" + f"/surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data_stageII"        + f"/iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))
    
    base_curves, base_currents = get_stageII_data(length=length, Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    base_currents = [Current(curr.x*4 * np.pi * 1e-7) for curr in base_currents]
    base_currents = [ScaledCurrent(curr, 1/(4 * np.pi * 1e-7)) for curr in base_currents]

    mr = np.sqrt(0.5678672050465505/ ( 2 * np.pi**2 ) )
    mR = np.sqrt(1.2 / (2 * np.pi**2) )
    vol_list = -2. * np.pi**2 * np.concatenate( (np.linspace(0.01,mr, 5), np.linspace( mr, mR, 6)[1:]) ) ** 2

    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    boozer_surface_list = []
    for idx in idx_surfaces:

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
        
        iota0 = iotaG[idx,0]
        G0 = iotaG[idx,1]
        
        bs = BiotSavart(coils)
        ll = Volume(s)
        
        # need to actually store target surface label for BoozerLS surfaces
        target = vol_list[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
        r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
        if verbose:
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={ll.J():.3f}, |label error|={np.abs(ll.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
        boozer_surface_list.append(boozer_surface)
    return boozer_surface_list, base_curves, base_currents, coils








def get_landreman_data(length=18):
    order=16
    ppp=10

    assert length == 18 or length == 20 or length == 22 or length == 24
    

    DIR = os.path.dirname(os.path.realpath(__file__))
    dofs = np.loadtxt(DIR + f'/data_landreman/len{length}.txt')
    currents = [1.] + dofs[:3].tolist()
    geo_dofs = dofs[3:].reshape((4, -1))

    base_currents = [ScaledCurrent(Current(c), 1e5) for c in currents]
    base_curves = [CurveXYZFourier(order*ppp, order) for i in range(4)]
    for i in range(4):
        base_curves[i].x = geo_dofs[i, :]

    return (base_curves, base_currents)


def compute_surfaces_in_landreman(mpol=10, ntor=10, exact=True, Nt_coils=16, write_to_file=False, vol_list=None, tol=1e-13, length=18):
    nfp = 2
    stellsym = True

    base_curves, base_currents = get_landreman_data(length=length)
    base_currents = [Current(curr.x) for curr in base_currents]
    base_currents = [ScaledCurrent(curr, 1e5) for curr in base_currents]
 
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    
    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), nquadphi, endpoint=False)
        thetas = np.linspace(0, 1, nquadtheta, endpoint=False)
    
    bs = BiotSavart(coils)
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    ma = find_magnetic_axis(bs, 60, 1.)
    s.fit_to_curve(ma, 0.05, flip_theta=True)
    

    iota0 = -0.39

    curr_sum = np.sum([curr.x for curr in base_currents]) * 1e5 
    G0 = -2. * np.pi * 2 * nfp * curr_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
    

    #vol_list = [-0.05, -0.187, -0.44, -0.5, -0.5915342336454275] 
    minor_R_list = np.linspace(0, np.sqrt(0.187/(2*np.pi**2)), 5)[1:].tolist() + np.linspace(np.sqrt(0.187/(2*np.pi**2)), np.sqrt(0.5915342336454275/(2*np.pi**2)), 5)[1:].tolist()
    vol_list = -np.pi*np.array(minor_R_list)**2. * 2 * np.pi

    boozer_surface_list = []
    boozer_surface_dict = []

    backup_dofs = s.get_dofs()
    backup_iota = iota0
    backup_G = G0

    #import ipdb;ipdb.set_trace()
    for idx,target in enumerate(vol_list):
        bs = BiotSavart(coils)
        label = Volume(s)
        boozer_surface = BoozerSurface(bs, s, label, target)
        try:
            s.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            if exact:
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', hessian=True)
                res['solver'] = 'LVM'
                if not res['success']:
                    boozer_surface.need_to_run_code = True
                    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=5e-10, maxiter=30, constraint_weight=100., iota=res['iota'], G=res['G'])
                    res['solver'] = 'NEWTON'
            
            if res['type'] == 'exact':
                print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e} ")
            elif res['solver'] == 'LVM':
                print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
            elif res['solver'] == 'NEWTON':
                print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}")
            
            #qp = np.linspace(0, 1, 100, endpoint=False)
            #snew = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=qp, quadpoints_theta=qp)
            #snew.x = s.x
            #print(snew.aspect_ratio())
            
            if res['success']:
                boozer_surface_list.append(boozer_surface)
                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G'], 'label': boozer_surface.surface.volume(), 'target': target})
                backup_dofs = s.get_dofs().copy()
                backup_iota = res['iota']
                backup_G = res['G']
                s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        except Exception as inst:
            print("Surface solver exception: ", type(inst))
    
    if write_to_file:
        DIR = os.path.dirname(os.path.realpath(__file__))
        import time
        ts = time.time()
        with open(DIR + "/data_landreman/" + f"surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
        with open(DIR + '/data_landreman/' +        f"iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s\n' % (surf_dict['iota']))
                f.write('%s\n' % (surf_dict['G']))
        with open(DIR + '/data_landreman/' +        f"voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s %s\n' % (surf_dict['target'], surf_dict['label']))

    return boozer_surface_list, base_curves, base_currents, coils

def load_surfaces_in_landreman(length=18, mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=np.arange(8), exact=True, time_stamp=None, tol=1e-13, verbose=False):
    nfp = 2
    stellsym = True
    if exact:
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1,   2*mpol+1, endpoint=False)
    else:
        phis = np.linspace(0, 1/(2*nfp), ntor+1+5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1+5, endpoint=False)

    nquadphi   = phis.size
    nquadtheta = thetas.size

    DIR = os.path.dirname(os.path.realpath(__file__))
    with open(DIR + "/data_landreman" + f"/surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data_landreman"        + f"/iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))
    
    base_curves, base_currents = get_landreman_data(length=length)
    base_currents = [Current(curr.x) for curr in base_currents]
    base_currents = [ScaledCurrent(curr, 1e5) for curr in base_currents]

    minor_R_list = np.linspace(0, np.sqrt(0.187/(2*np.pi**2)), 5)[1:].tolist() + np.linspace(np.sqrt(0.187/(2*np.pi**2)), np.sqrt(0.5915342336454275/(2*np.pi**2)), 5)[1:].tolist()
    vol_list = -np.pi*np.array(minor_R_list)**2. * 2 * np.pi
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    boozer_surface_list = []
    for idx in idx_surfaces:

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
        
        iota0 = iotaG[idx, 0]
        G0 = iotaG[idx, 1]
        
        bs = BiotSavart(coils)
        ll = Volume(s)
        
        # need to actually store target surface label for BoozerLS surfaces
        target = vol_list[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        #res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
        if exact:
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
            r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={label.J():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(r, ord=np.inf):.3e}, cond = {np.linalg.cond(res['jacobian']):.3e}")
        else:
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-13, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', hessian=True)
            res['solver'] = 'LVM'
            if not res['success']:
                boozer_surface.need_to_run_code = True
                res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=5e-13, maxiter=30, constraint_weight=100., iota=res['iota'], G=res['G'])
                res['solver'] = 'NEWTON'
        
        if res['type'] == 'exact':
            print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e} ")
        elif res['solver'] == 'LVM':
            print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
        elif res['solver'] == 'NEWTON':
            print(f"{res['solver']}     iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}")
        
        boozer_surface_list.append(boozer_surface)
    return boozer_surface_list, base_curves, base_currents, coils











def is_self_intersecting(cs):
    """
    This function takes as input a cross section, represented as a polygon.
    """
    R = np.sqrt( cs[:,0]**2 + cs[:,1]**2)
    Z = cs[:, 2]

    context = get_context()
    Point, Contour = context.point_cls, context.contour_cls
    contour = Contour([ Point(R[i], Z[i]) for i in range(cs.shape[0]) ])
    return contour_self_intersects(contour)


#def compute_surfaces_continuation(boozer_surface, coils, vol_list, add_res_list=[], tol=1e-13, weight=100., maxiter=60):
#
#    backup_dofs = boozer_surface.surface.get_dofs()
#    backup_iota = boozer_surface.res['iota']
#    backup_G = boozer_surface.res['G']
#  
#    res_list = ['label', 'ratio', 'iota'] + add_res_list
#    res_dict={}
#    for r in res_list:
#        res_dict[r]=[]
#
#    for idx,target in enumerate(vol_list):
#        try:
#            boozer_surface.surface.set_dofs(backup_dofs)
#            iota0 = backup_iota
#            G0 = backup_G
#
#            boozer_surface.targetlabel = target
#            if boozer_surface.res['type'] == 'exact':
#                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=maxiter,iota=iota0,G=G0)
#                r, = boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=0)
#                res_norm= np.linalg.norm(r, ord=np.inf)
#                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={boozer_surface.label.J():.3f}, |rel. label error|={np.abs(boozer_surface.label.J()-target)/np.abs(target):.3e}, ||residual||_inf={res_norm:.3e}")
#            else:
#                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=maxiter, constraint_weight=weight, iota=iota0, G=G0, method='manual')
#                # if LM didn't work, try Newton 
#                if not res['success']:
#                    print("failed, trying Newton")
#                    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=tol, maxiter=maxiter, constraint_weight=weight, iota=res['iota'], G=res['G'])
#                r, J= boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=1)
#                res_norm = np.linalg.norm(r)
#                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={boozer_surface.label.J():.3f}, |rel. label error|={np.abs(boozer_surface.label.J()-target)/np.abs(target):.3e}, ||residual||_2={np.linalg.norm(r):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
#            
#            if not res['success']:
#                continue
#            if is_self_intersecting(boozer_surface.surface.cross_section(0, thetas=200)):
#                print("Surface is self-intersecting!")
#           
#
#            backup_dofs = boozer_surface.surface.get_dofs().copy()
#            backup_iota = res['iota']
#            backup_G = res['G']
#            
#            def compute_non_quasisymmetry_L2(in_surface, in_coils):
#                bs = BiotSavart(in_coils)
#                phis = np.linspace(0, 1/in_surface.nfp, 200, endpoint=False)
#                thetas = np.linspace(0, 1., 200, endpoint=False)
#                s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
#                s.set_dofs(in_surface.get_dofs())
#
#                x = s.gamma()
#                B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
#                mod_B = np.linalg.norm(B, axis=2)
#                n = np.linalg.norm(s.normal(), axis=2)
#                mean_phi_mod_B = np.mean(mod_B*n, axis=0)/np.mean(n, axis=0)
#                mod_B_QS = mean_phi_mod_B[None, :]
#                mod_B_non_QS = mod_B - mod_B_QS
#                non_qs = np.mean(mod_B_non_QS**2 * n)**0.5
#                qs = np.mean(mod_B_QS**2 * n)**0.5
#                return non_qs, qs, s.area(), mod_B 
#            
#            def compute_tf(in_surface, in_coils):
#                bs_tf = BiotSavart(in_coils)
#                phis = np.linspace(0, 1/in_surface.nfp, 200, endpoint=False)
#                thetas = np.linspace(0, 1., 200, endpoint=False)
#                s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
#                s.set_dofs(in_surface.get_dofs())
#                tf = ToroidalFlux(s, bs_tf)
#                return tf.J()
#
#            
#            non_qs, qs, area, absB = compute_non_quasisymmetry_L2(boozer_surface.surface, coils)
#
#            res_dict['label'].append( np.abs(boozer_surface.label.J() ))
#            res_dict['iota'].append( np.abs(boozer_surface.res['iota']))
#            res_dict['ratio'].append(non_qs/qs)
#            if 'res' in res_dict:
#                res_dict['res'].append(res_norm)
#            if 'dofs' in res_dict:
#                res_dict['dofs'].append( boozer_surface.surface.get_dofs() )
#            if 'tf' in res_dict:
#                tf = compute_tf(boozer_surface.surface, coils)
#                res_dict['tf'].append(tf)
#            if 'absB' in res_dict:
#                res_dict['absB'].append(absB)
#            if 'ratio_nonqs_area' in res_dict:
#                res_dict['ratio_nonqs_area'].append(non_qs/(area**0.5))
#        except Exception as e:
#            print("Didn't converge", e)
#    return res_dict

def compute_surfaces_continuation(boozer_surface, coils, label, add_res_list=[], tol=1e-13, weight=100., maxiter=60):

    backup_dofs = boozer_surface.surface.get_dofs()
    backup_iota = boozer_surface.res['iota']
    backup_G = boozer_surface.res['G']
  
    res_list = ['label', 'ratio', 'iota'] + add_res_list
    res_dict={}
    for r in res_list:
        res_dict[r]=[]
    
    if type(label) is list:
        num = len(label)
    else:
        num = label

    for idx in range(num):
        try:
            boozer_surface.surface.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G
            
            if type(label) is list:
                boozer_surface.targetlabel = label[idx]
            else:
                boozer_surface.surface.extend_via_normal(-0.005)
                boozer_surface.targetlabel = Volume(boozer_surface.surface).J()
                target = Volume(boozer_surface.surface).J()

            if boozer_surface.res['type'] == 'exact':
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=maxiter,iota=iota0,G=G0)
                r, = boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=0)
                res_norm= np.linalg.norm(r, ord=np.inf)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={boozer_surface.label.J():.3f}, |rel. label error|={np.abs(boozer_surface.label.J()-target)/np.abs(target):.3e}, ||residual||_inf={res_norm:.3e}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=maxiter, constraint_weight=weight, iota=iota0, G=G0, method='manual')
                # if LM didn't work, try Newton 
                if not res['success']:
                    print("failed, trying Newton")
                    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=tol, maxiter=maxiter, constraint_weight=weight, iota=res['iota'], G=res['G'])
                r, J= boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=1)
                res_norm = np.linalg.norm(r)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={boozer_surface.label.J():.3f}, |rel. label error|={np.abs(boozer_surface.label.J()-target)/np.abs(target):.3e}, ||residual||_2={np.linalg.norm(r):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
            
            if not res['success']:
                continue
            if is_self_intersecting(boozer_surface.surface.cross_section(0, thetas=200)):
                print("Surface is self-intersecting!")
           

            backup_dofs = boozer_surface.surface.get_dofs().copy()
            backup_iota = res['iota']
            backup_G = res['G']
            
            def compute_non_quasisymmetry_L2(in_surface, in_coils):
                bs = BiotSavart(in_coils)
                phis = np.linspace(0, 1/in_surface.nfp, 200, endpoint=False)
                thetas = np.linspace(0, 1., 200, endpoint=False)
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
                return non_qs, qs, s.area(), mod_B 
            
            def compute_tf(in_surface, in_coils):
                bs_tf = BiotSavart(in_coils)
                phis = np.linspace(0, 1/in_surface.nfp, 200, endpoint=False)
                thetas = np.linspace(0, 1., 200, endpoint=False)
                s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
                s.set_dofs(in_surface.get_dofs())
                tf = ToroidalFlux(s, bs_tf)
                return tf.J()

            
            non_qs, qs, area, absB = compute_non_quasisymmetry_L2(boozer_surface.surface, coils)

            res_dict['label'].append( np.abs(boozer_surface.label.J() ))
            res_dict['iota'].append( np.abs(boozer_surface.res['iota']))
            res_dict['ratio'].append(non_qs/qs)
            if 'res' in res_dict:
                res_dict['res'].append(res_norm)
            if 'dofs' in res_dict:
                res_dict['dofs'].append( boozer_surface.surface.get_dofs() )
            if 'tf' in res_dict:
                tf = compute_tf(boozer_surface.surface, coils)
                res_dict['tf'].append(tf)
            if 'absB' in res_dict:
                res_dict['absB'].append(absB)
            if 'ratio_nonqs_area' in res_dict:
                res_dict['ratio_nonqs_area'].append(non_qs/(area**0.5))
        except Exception as e:
            print("Didn't converge", e)
    return res_dict

def find_magnetic_axis(biotsavart, n, rguess):
    from scipy.spatial.distance import cdist
    from scipy.optimize import fsolve
    points = np.linspace(0, 2*np.pi, n, endpoint=False).reshape((n, 1))
    oneton = np.asarray(range(0, n)).reshape((n, 1))
    fak = 2*np.pi / (points[-1] - points[0] + (points[1]-points[0]))
    dists = fak * cdist(points, points, lambda a, b: a-b)
    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    if n % 2 == 0:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.tan(0.5 * dists)
    else:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.sin(0.5 * dists)

    np.fill_diagonal(D, 0)
    D *= fak
    phi = points

    def build_residual(rz):
        inshape = rz.shape
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        Br = np.cos(phi)*Bx + np.sin(phi)*By
        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
        residual_r = D @ r - r * Br / Bphi
        residual_z = D @ z - r * Bz / Bphi
        return np.vstack((residual_r, residual_z)).reshape(inshape)

    def build_jacobian(rz):
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        GradB = biotsavart.dB_by_dX()
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        dxBx = GradB[:, 0, 0].reshape((n, 1))
        dyBx = GradB[:, 1, 0].reshape((n, 1))
        dzBx = GradB[:, 2, 0].reshape((n, 1))
        dxBy = GradB[:, 0, 1].reshape((n, 1))
        dyBy = GradB[:, 1, 1].reshape((n, 1))
        dzBy = GradB[:, 2, 1].reshape((n, 1))
        dxBz = GradB[:, 0, 2].reshape((n, 1))
        dyBz = GradB[:, 1, 2].reshape((n, 1))
        dzBz = GradB[:, 2, 2].reshape((n, 1))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        Br = cosphi*Bx + sinphi*By
        Bphi = cosphi*By - sinphi*Bx
        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
        dzBr = cosphi*dzBx + sinphi*dzBy
        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
        dzBphi = cosphi*dzBy - sinphi*dzBx
        drBz = cosphi * dxBz + sinphi*dyBz
        # residual_r = D @ r - r * Br / Bphi
        # residual_z = D @ z - r * Bz / Bphi
        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])
    
    r0 = np.ones_like(phi) * rguess
    z0 = np.zeros_like(phi)
    x0 = np.vstack((r0, z0))
    # h = np.random.rand(x0.size).reshape(x0.shape)
    # eps = 1e-4
    # drdh = build_jacobian(x0)@h
    # drdh_est = (build_residual(x0+eps*h)-build_residual(x0-eps*h))/(2*eps)
    # err = np.linalg.norm(drdh-drdh_est)
    # print(err)
    # print(np.hstack((drdh, drdh_est)))

    # diff = 1e10
    # soln = x0.copy()
    # for i in range(50):
        # r = build_residual(soln)
        # print("r", np.linalg.norm(r))
        # update = np.linalg.solve(build_jacobian(soln), r)
        # soln -= 0.01 * update
        # diff = np.linalg.norm(update)
        # print('dx', diff)

    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)
    #rz = np.hstack((soln[:n, None], phi, soln[n:, None]))
    xyz = np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))
    quadpoints = np.linspace(0, 1, n, endpoint=False)
    ma = CurveRZFourier(quadpoints, n//2, 1, True)
    ma.least_squares_fit(xyz)
    dofs = ma.x.copy()

    quadpoints = np.linspace(0, 1/2, n, endpoint=False)
    ma = CurveRZFourier(quadpoints, n//2, 1, True)
    ma.x = dofs
    
    ma2 = CurveRZFourier(quadpoints, n//2, 2, True)
    ma2.least_squares_fit(ma.gamma())
    
    return ma2
