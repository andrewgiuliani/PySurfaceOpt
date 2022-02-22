import numpy as np
import os
from simsopt.util.zoo import get_ncsx_data
from simsopt.field.coil import ScaledCurrent, Current, coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.boozersurface import BoozerSurface, boozer_surface_residual
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from pysurfaceopt.surfaceobjectives import Volume



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
    G0 = 2. * np.pi * curr_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
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

    base_curves, base_currents, ma = get_stageII_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
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



def compute_surfaces_in_stageII(mpol=10, ntor=10, exact=True, Nt_coils=12, write_to_file=False, vol_list=None, tol=1e-13, length=18):
    nfp = 2
    stellsym = True


    PPP = 10
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

def load_surfaces_in_stageII(length=18, mpol=10, ntor=10, stellsym=True, Nt_coils=6, idx_surfaces=np.arange(10), exact=True, time_stamp=None, tol=1e-13):
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




