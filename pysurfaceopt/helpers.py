import numpy as np
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import NonQuasiAxisymmetricComponent, Volume, ToroidalFlux, boozer_surface_residual
from simsopt.geo.boozersurface import BoozerSurface
import os

def get_ncsx_data(Nt_coils=25, Nt_ma=10, ppp=10):
    DIR = os.path.dirname(os.path.realpath(__file__))
    filename = DIR + '/NCSX_coil_coeffs.dat'
    coils = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
    nfp = 3
    currents = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]
    cR = [
        1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439,
        -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05,
        2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06,
        -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08,
        3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11,
        1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12,
        -1.057727690156884e-12, 5.253991686160475e-13]
    sZ = [
        0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824,
        -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06,
        2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07,
        -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09,
        2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12,
        1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13,
        -6.897549209312209e-14]

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
    ma.rc[:] = cR[0:(Nt_ma+1)]
    ma.zs[:] = sZ[0:Nt_ma]
    return (coils, currents, ma)

def get_florian_data(Nt_coils=16, Nt_ma=10, ppp=20, length=18):
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
    c0 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_0.txt')
    c1 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_1.txt')
    c2 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_2.txt')
    c3 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_current_3.txt')
    currents = [ c0, c1, c2, c3 ]
    currents = [float(c)*1e5 for c in currents]

    dofs0 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_0.txt')
    dofs1 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_1.txt')
    dofs2 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_2.txt')
    dofs3 = np.loadtxt(DIR + f'/data_florian/coils_from_florian/output_well_False_lengthbound_{length}.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_{ig}_order_16_expquad_qfm_None_curve_3.txt')


    cR = np.array([])
    sZ = np.array([])


    cR = np.array([1.0036926511470303e+00,  1.8407050366440988e-01,
        2.1754740443305290e-02,  2.6126268948202681e-03,
        3.1094019923834750e-04,  3.5615840692013363e-05,
        1.1575557931107334e-05, -9.7525514681800379e-06,
        5.4555286892534256e-07,  9.5233296031365272e-06,
        8.1100972280486715e-06,  4.1222798726227710e-06,
        1.5776932093403984e-06,  4.9337672641195391e-07,
        1.3502817067043504e-07,  3.3089890428149458e-08,
        4.6651654527380966e-09, -8.9242804612548248e-10,
        1.8730468999460285e-10,  1.3606296925676505e-09,
        1.3679738331443444e-09,  8.6234514683950866e-10,
        4.1246297708529286e-10,  1.6057127584761418e-10,
        5.2291481174502994e-11,  1.4242963338298721e-11])
    sZ = np.array([ 1.5821038112098335e-01,  2.0665891596217307e-02,
        2.5785292044606099e-03,  3.1422101155062558e-04,
        3.8816882106881092e-05,  5.9841437303069453e-06,
       -1.3394733868103493e-06,  6.0525252567547796e-07,
        2.1544297427248749e-06,  2.1953122045881845e-06,
        1.3554347958960682e-06,  6.0087287560948415e-07,
        2.2012331361766568e-07,  6.2422468531104817e-08,
        1.6692354474428135e-08,  5.1324453142511082e-09,
        1.2690858057717800e-09,  4.6446231228333925e-10,
        6.6530065091559781e-10,  7.2649041259036759e-10,
        5.3067841500065600e-10,  2.9348890911607923e-10,
        1.3223038469946057e-10,  5.0382881348380751e-11,
        1.6576074623208554e-11])

    nfp = 2
    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
    ma.rc[:] = cR[0:(Nt_ma+1)]
    ma.zs[:] = sZ[0:Nt_ma]

    coils = [CurveXYZFourier(order*ppp, order) for i in range(4)]
    coils[0].set_dofs(dofs0)
    coils[1].set_dofs(dofs1)
    coils[2].set_dofs(dofs2)
    coils[3].set_dofs(dofs3)

    return (coils, currents, ma)

def compute_surfaces_florian(mpol=10, ntor=10, exact=True, Nt_coils=16, length=18, write_to_file = False):
    
    DIR = os.path.dirname(os.path.realpath(__file__))
    coils, currents, ma = get_florian_data(length=length)
    stellarator = CoilCollection(coils, currents, 2, True)
    
    stellsym = True
    nfp = 2
 
    coils = stellarator.coils
    currents = stellarator.currents


    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), ntor + 1 + 5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol + 1 + 5, endpoint=False)

    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    DIR = os.path.dirname(os.path.realpath(__file__))
    past_dofs = np.loadtxt(DIR + '/data_florian/' +'surface_dofs_mpol=10_ntor=10_nquadphi=11_nquadtheta=21_stellsym=True_exact=True_Nt_coils=6.txt') 
    s.set_dofs(past_dofs[0,:])

    past_iotaG = np.loadtxt(DIR + '/data_florian/' +'iotaG_mpol=10_ntor=10_nquadphi=11_nquadtheta=21_stellsym=True_exact=True_Nt_coils=6.txt') 
    iota0 = past_iotaG[0]

    #from mayavi import mlab
    #for i in range(0, len(coils)):
    #    g = coils[i].gamma()
    #    mlab.plot3d(g[:, 0], g[:, 1], g[:, 2])
    #gamma = s.gamma()
    #mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2])
    #mlab.show()


    mr = np.sqrt(0.5678672050465505/ ( 2 * np.pi**2 ) )
    mR = np.sqrt(1.2 / (2 * np.pi**2) )
    vol_list = -2. * np.pi**2 * np.concatenate( (np.linspace(0.01,mr, 5), np.linspace( mr, mR, 6)[1:]) ) ** 2
    G0 = 2. * np.pi * np.sum(np.abs(currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    boozer_surface_list = []
    boozer_surface_dict = []

    backup_dofs = s.get_dofs()
    backup_iota = iota0
    backup_G = G0

    nfp = 2
    nphi = 32
    ntheta = 32
    phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
    phis+=phis[1]/2
    thetas = np.linspace(0, 1., ntheta, endpoint=False)
    filename = DIR + "/data_florian/input.LandremanPaul2021_QA" 
    ref_surf = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
    print(ref_surf.volume())
    print(ref_surf.aspect_ratio())

    ar_surf = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    
    for idx,target in enumerate(vol_list):
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Volume(s, stellarator)
        boozer_surface = BoozerSurface(bs, s, label, target)
        
        try:
            s.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G
            
            if exact:
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=iota0,G=G0)
                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
                tf = ToroidalFlux(s, bs_tf, stellarator)
                ar_surf.set_dofs(boozer_surface.surface.get_dofs())
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}, aspect_ratio = {ar_surf.aspect_ratio()}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
                tf = ToroidalFlux(s, bs_tf, stellarator)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")

            if res['success']:
                boozer_surface_list.append(boozer_surface)
                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G']})
                backup_dofs = s.get_dofs().copy()
                backup_iota = res['iota']
                backup_G = res['G']
            
            #import matplotlib.pyplot as plt
            #ax = plt.axes(projection='3d')
            #for i in range(0, len(stellarator.coils)):
            #    ax = stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(stellarator._base_coils)])
            #ax.view_init(elev=90., azim=0)
            #ax.set_xlim(-2, 2)
            #ax.set_ylim(-2, 2)
            #ax.set_zlim(-1, 1)

            #import mayavi.mlab as mlab
            #gamma = boozer_surface.surface.gamma()
            #gamma_ref = ref_surf.gamma()
            #mlab.mesh(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2])
            #mlab.mesh(gamma_ref[:,:,0], gamma_ref[:,:,1], gamma_ref[:,:,2])

        except:
            print("Didn't converge")
    
    #mlab.show()
    if write_to_file:
        import time
        ts = time.time()
        with open(DIR + "/data_florian/" + f"surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
        with open(DIR + '/data_florian/' +        f"iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s\n' % (surf_dict['iota']))
                f.write('%s\n' % (surf_dict['G']))
    return boozer_surface_list, coils, currents, ma, stellarator

def load_surfaces_florian(mpol=10, ntor=10, stellsym=True, Nt_coils=16, idx_surfaces=np.arange(10), exact=True, length=18, time_stamp=None):
    nfp = 2
    nphi = 32
    ntheta = 32
    phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
    phis+=phis[1]/2
    thetas = np.linspace(0, 1., ntheta, endpoint=False)
    ar_surf = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)


    stellsym = True
    nfp = 2

    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), ntor + 1 + 5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol + 1 + 5, endpoint=False)

    DIR = os.path.dirname(os.path.realpath(__file__))
    with open(DIR + "/data_florian" + f"/surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data_florian"        + f"/iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_length={length}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))
 
    coils, currents, ma = get_florian_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=10, length=length)
    stellarator = CoilCollection(coils, currents, nfp, True)
   
    mr = np.sqrt(0.5678672050465505/ ( 2 * np.pi**2 ) )
    mR = np.sqrt(1.2 / (2 * np.pi**2) )
    vol_list = -2. * np.pi**2 * np.concatenate( (np.linspace(0.01,mr, 5), np.linspace( mr, mR, 6)[1:]) ) ** 2

    boozer_surface_list = []
    for idx in idx_surfaces:

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
       
        iota0 = iotaG[idx,0]
        G0 = iotaG[idx,1]

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        tf = ToroidalFlux(s, bs_tf, stellarator)

        ll = Volume(s, stellarator)
        target = vol_list[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        if exact:
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=iota0,G=G0)
            bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
            tf = ToroidalFlux(s, bs_tf, stellarator)
            ar_surf.set_dofs(boozer_surface.surface.get_dofs())
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}, aspect_ratio = {ar_surf.aspect_ratio()}")
        else:
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
            bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
            tf = ToroidalFlux(s, bs_tf, stellarator)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, ||residual||_2={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
        boozer_surface_list.append(boozer_surface)
    return boozer_surface_list, coils, currents, ma, stellarator




def compute_surfaces_continuation(boozer_surface, stellarator, vol_list, phi=None, tol=1e-13):

    backup_dofs = boozer_surface.surface.get_dofs()
    backup_iota = boozer_surface.res['iota']
    backup_G = boozer_surface.res['G']

    label_list = []
    ratio_list = []
    iota_list = []
    res_list = []
    b_dot_n_list = []
    xs_list = []

    def is_self_intersecting(in_s):
        """
        This function takes as input a cross section, represented as a polygon.
        """
        cs = in_s.cross_section(0.)
        R = np.sqrt( cs[:,0]**2 + cs[:,1]**2)
        Z = cs[:, 2]
        
        from ground.base import get_context
        context = get_context()
        Point, Contour = context.point_cls, context.contour_cls
        contour = Contour([ Point(R[i], Z[i]) for i in range(cs.shape[0]) ])
        from bentley_ottmann.planar import contour_self_intersects
        return contour_self_intersects(contour)




    for idx,target in enumerate(vol_list):
        try:
            boozer_surface.surface.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            boozer_surface.targetlabel = target
            if boozer_surface.res['type'] == 'exact':
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
                r, = boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=0)
                res_norm= np.linalg.norm(r, ord=np.inf)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={boozer_surface.surface.volume():.3f}, |label error|={np.abs(boozer_surface.label.J()-target):.3e}, ||residual||_inf={res_norm:.3e}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                r, J= boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=1)
                res_norm = np.linalg.norm(r)
                #res_norm = np.linalg.norm(J.T @ r, ord=np.inf)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, vol={boozer_surface.surface.volume():.3f}, |label error|={np.abs(boozer_surface.label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(boozer_surface.surface, res['iota'], res['G'], boozer_surface.bs, derivatives=0)[0], ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")

            if res['success']:
                if is_self_intersecting(boozer_surface.surface):
                    print("BoozerLS is self-intersecting!")
                else: 
                    label_list.append( np.abs(boozer_surface.label.J() ))
                    iota_list.append( np.abs(boozer_surface.res['iota']))
                    backup_dofs = boozer_surface.surface.get_dofs().copy()
                    backup_iota = res['iota']
                    backup_G = res['G']
 

                    def Bdotn(in_surface, bs):
                        sDIM = 50
                        phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
                        thetas = np.linspace(0, 1., sDIM, endpoint=False)
                        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
                        s.set_dofs(in_surface.get_dofs())
 
                        x = s.gamma()
                        B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
                        n = np.linalg.norm(s.normal(), axis=2)
                        absB = np.linalg.norm(B, axis=2)
                        bdotn = (np.sum(B*s.normal(), axis=2)/absB)**2
                        return np.mean(bdotn*n)/s.area()
                    
                    def compute_non_quasisymmetry_L2(s, bs):
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
                    non_qs, qs = compute_non_quasisymmetry_L2(boozer_surface.surface, boozer_surface.bs)
                    ratio_list.append(non_qs/qs)
                    res_list.append(res_norm)
                    bdotn = Bdotn(boozer_surface.surface, boozer_surface.bs)
                    b_dot_n_list.append( bdotn )
                    if phi is not None:
                        xs_list.append(boozer_surface.surface.cross_section(phi, thetas=1000))
        except Exception as e:
            print("Didn't converge", e)
    if phi is None:
        return label_list, ratio_list, iota_list, res_list, b_dot_n_list
    return label_list, ratio_list, iota_list, res_list, b_dot_n_list, xs_list

def compute_surfaces_continuation_both(boozerLS, boozerExact, vol_list):

    backup_dofs = boozerLS.surface.get_dofs()
    backup_iota = boozerLS.res['iota']
    backup_G    = boozerLS.res['G']

    label_listLS = []
    ratio_listLS = []
    iota_listLS = []


    label_listExact = []
    ratio_listExact = []
    iota_listExact = []



    def compute_non_quasisymmetry_L2(s, bs):
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
 
    def is_self_intersecting(in_s):
        """
        This function takes as input a cross section, represented as a polygon.
        """
        cs = in_s.cross_section(0.)
        R = np.sqrt( cs[:,0]**2 + cs[:,1]**2)
        Z = cs[:, 2]
        
        from ground.base import get_context
        context = get_context()
        Point, Contour = context.point_cls, context.contour_cls
        contour = Contour([ Point(R[i], Z[i]) for i in range(cs.shape[0]) ])
        from bentley_ottmann.planar import contour_self_intersects
        out = contour_self_intersects(contour)
        #if out :
        #    import matplotlib.pyplot as plt
        #    plt.plot(R, Z)
        #    plt.show()
        #    import ipdb;ipdb.set_trace()
        return out



    for idx,target in enumerate(vol_list):
        try:
            boozerLS.surface.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            boozerLS.targetlabel = target
            resLS = boozerLS.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
            r, J = boozer_surface_residual(boozerLS.surface, resLS['iota'], resLS['G'], boozerLS.bs, derivatives=1)
            res_inf = np.linalg.norm(J.T @ r, ord=np.inf)
            print(f"iter={resLS['iter']}, iota={resLS['iota']:.16f}, vol={boozerLS.label.J():.3f}, |label error|={np.abs(boozerLS.label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(boozerLS.surface, resLS['iota'], resLS['G'], boozerLS.bs, derivatives=0)[0], ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(resLS['gradient'], ord=np.inf):.3e}")

            boozerExact.surface.set_dofs(boozerLS.surface.get_dofs())
            boozerExact.targetlabel = target
            resExact = boozerExact.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=resLS['iota'],G=resLS['G'])
            r, = boozer_surface_residual(boozerExact.surface, resExact['iota'], resExact['G'], boozerExact.bs, derivatives=0)
            res_inf = np.linalg.norm(r, ord=np.inf)
            print(f"iter={resExact['iter']}, iota={resExact['iota']:.16f}, vol={boozerExact.label.J():.3f}, |label error|={np.abs(boozerExact.label.J()-target):.3e}, ||residual||_inf={res_inf:.3e}")

         
            if resLS['success']:
                if is_self_intersecting(boozerLS.surface):
                    print("BoozerLS is self-intersecting!")
                else: 
                    label_listLS.append( np.abs(boozerLS.label.J() ))
                    iota_listLS.append( np.abs(boozerLS.res['iota']))
                    backup_dofs = boozerLS.surface.get_dofs().copy()
                    backup_iota = resLS['iota']
                    backup_G = resLS['G']

                    non_qs, qs = compute_non_quasisymmetry_L2(boozerLS.surface, boozerLS.bs)
                    ratio_listLS.append(non_qs/qs)
            
            if resExact['success']:
                if is_self_intersecting(boozerExact.surface):
                    print("BoozerExact is self-intersecting!")
                else: 
                    label_listExact.append( np.abs(boozerExact.label.J() ))
                    iota_listExact.append( np.abs(boozerExact.res['iota']))

                    non_qs, qs = compute_non_quasisymmetry_L2(boozerExact.surface, boozerExact.bs)
                    ratio_listExact.append(non_qs/qs)
            #import ipdb;ipdb.set_trace()
        except:
            print("Didn't converge")
    return (label_listLS, ratio_listLS, iota_listLS), (label_listExact, ratio_listExact, iota_listExact)




# default tol for BoozerLS is 1e-10
# default tol for BoozerExact is 1e-13
def compute_surfaces_in_NCSX_half(mpol=10, ntor=10, exact=True, Nt_coils=6, write_to_file = False, vol_list=None, tol=1e-13):
    PPP = 20
    DIR = os.path.dirname(os.path.realpath(__file__))
    coils, currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    stellarator = CoilCollection(coils, currents, 3, True)

    coils = stellarator.coils
    currents = stellarator.currents
    
    stellsym = True
    nfp = 3
    
    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)[:ntor+1]
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), ntor + 1 + 5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol + 1 + 5, endpoint=False)
    
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.10, flip_theta=True)
    iota0 = -0.4
    
    if vol_list is None:
        vol_list = np.linspace(-0.162, -3.3, 10)
    
    G0 = 2. * np.pi * np.sum(np.abs(currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
    boozer_surface_list = []
    boozer_surface_dict = []

    backup_dofs = s.get_dofs()
    backup_iota = iota0
    backup_G = G0

    for idx,target in enumerate(vol_list):
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Volume(s, stellarator)
        boozer_surface = BoozerSurface(bs, s, label, target)
        
        try:
            s.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            if exact:
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=tol, maxiter=30,iota=iota0,G=G0)
                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
                tf = ToroidalFlux(s, bs_tf, stellarator)
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                res_inf = np.linalg.norm(r, ord=np.inf)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={res_inf:.3e}, cond = {np.linalg.cond(res['jacobian'])}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=tol, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
                tf = ToroidalFlux(s, bs_tf, stellarator)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")

            if res['success']:
                boozer_surface_list.append(boozer_surface)
                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G'], 'label': boozer_surface.surface.volume(), 'target': target})
                backup_dofs = s.get_dofs().copy()
                backup_iota = res['iota']
                backup_G = res['G']
                s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        except:
            print("Didn't converge")

    if write_to_file:
        import time
        ts = time.time()
        with open(DIR + "/data/" + f"half_surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
        with open(DIR + '/data/' +        f"half_iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s\n' % (surf_dict['iota']))
                f.write('%s\n' % (surf_dict['G']))
        with open(DIR + '/data/' +        f"half_voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s %s\n' % (surf_dict['target'], surf_dict['label']))

    return boozer_surface_list, coils, currents, ma, stellarator

def load_surfaces_in_NCSX_half(mpol=10, ntor=10, stellsym=True, Nt_coils=6, idx_surfaces=np.arange(10), exact=True, time_stamp=None):
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
    with open(DIR + "/data" + f"/half_surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data"        + f"/half_iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))
    with open(DIR + '/data' + f"/half_voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt") as f:
        inlabels = np.loadtxt(f)
        vol_targets = inlabels[:,0]
    coils, currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    stellarator = CoilCollection(coils, currents, 3, True)

    boozer_surface_list = []
    for idx in idx_surfaces:

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
        
        iota0 = iotaG[idx,0]
        G0 = iotaG[idx,1]

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        tf = ToroidalFlux(s, bs_tf, stellarator)
        ll = Volume(s, stellarator)

        # need to actually store target surface label for BoozerLS surfaces
        target = vol_targets[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        if exact:
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=iota0,G=G0)
            bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
            tf = ToroidalFlux(s, bs_tf, stellarator)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
        else:
#            import ipdb;ipdb.set_trace()
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', hessian=False)
            bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
            tf = ToroidalFlux(s, bs_tf, stellarator)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
        
        boozer_surface_list.append(boozer_surface)
    return boozer_surface_list, coils, currents, ma, stellarator





def compute_surfaces_in_NCSX(mpol=10, ntor=10, exact=True, Nt_coils=6, write_to_file = False, vol_list=None):
    PPP = 20
    DIR = os.path.dirname(os.path.realpath(__file__))
    coils, currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    stellarator = CoilCollection(coils, currents, 3, True)

    coils = stellarator.coils
    currents = stellarator.currents
    
    stellsym = True
    nfp = 3
    
    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
        phis = np.linspace(0, 1/(2*nfp), ntor + 1 + 5, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol + 1 + 5, endpoint=False)
    
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.10, flip_theta=True)
    iota0 = -0.4
    
    if vol_list is None:
        vol_list = np.linspace(-0.162, -3.3, 10)
    
    G0 = 2. * np.pi * np.sum(np.abs(currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
    boozer_surface_list = []
    boozer_surface_dict = []

    backup_dofs = s.get_dofs()
    backup_iota = iota0
    backup_G = G0

    for idx,target in enumerate(vol_list):
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        label = Volume(s, stellarator)
        boozer_surface = BoozerSurface(bs, s, label, target)
        
        try:
            s.set_dofs(backup_dofs)
            iota0 = backup_iota
            G0 = backup_G

            if exact:
                res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=iota0,G=G0)
                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
                tf = ToroidalFlux(s, bs_tf, stellarator)
                r, = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)
                res_inf = np.linalg.norm(r, ord=np.inf)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={res_inf:.3e}, cond = {np.linalg.cond(res['jacobian'])}")
                #print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={res_inf:.3e}")
            else:
                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
                tf = ToroidalFlux(s, bs_tf, stellarator)
                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")

            if res['success']:
                boozer_surface_list.append(boozer_surface)
                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G'], 'label': boozer_surface.surface.volume(), 'target': target})
                backup_dofs = s.get_dofs().copy()
                backup_iota = res['iota']
                backup_G = res['G']
                s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        except:
            print("Didn't converge")

    if write_to_file:
        import time
        ts = time.time()
        with open(DIR + "/data/" + f"surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
        with open(DIR + '/data/' +        f"iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s\n' % (surf_dict['iota']))
                f.write('%s\n' % (surf_dict['G']))
        with open(DIR + '/data/' +        f"voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{ts}.txt", 'w') as f:
            for surf_dict in boozer_surface_dict:
                f.write('%s %s\n' % (surf_dict['target'], surf_dict['label']))

    return boozer_surface_list, coils, currents, ma, stellarator





def load_surfaces_in_NCSX(mpol=10, ntor=10, stellsym=True, Nt_coils=6, idx_surfaces=np.arange(10), exact=True, time_stamp=None):
    nfp = 3    
    if exact:
        nquadphi = ntor+1
        nquadtheta = 2*mpol+1
    else:
        nquadphi = ntor + 1 + 5
        nquadtheta = 2*mpol + 1 + 5
     
    phis = np.linspace(0, 1/(2*nfp), nquadphi, endpoint=False)
    thetas = np.linspace(0, 1, nquadtheta, endpoint=False)

    PPP = 20
    DIR = os.path.dirname(os.path.realpath(__file__))
    with open(DIR + "/data" + f"/surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt", 'r') as f:
        dofs = np.loadtxt(f)
    with open(DIR + "/data"        + f"/iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt", 'r') as f:
        iotaG = np.loadtxt(f).reshape((-1,2))
    with open(DIR + '/data' + f"/voltargets_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}_PPP={PPP}_{time_stamp}.txt") as f:
        inlabels = np.loadtxt(f)
        vol_targets = inlabels[:,0]
    coils, currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=PPP)
    stellarator = CoilCollection(coils, currents, 3, True)

    boozer_surface_list = []
    for idx in idx_surfaces:

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(dofs[idx,:])
        
        iota0 = iotaG[idx,0]
        G0 = iotaG[idx,1]

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        tf = ToroidalFlux(s, bs_tf, stellarator)
        ll = Volume(s, stellarator)

        # need to actually store target surface label for BoozerLS surfaces
        target = vol_targets[idx]
        boozer_surface = BoozerSurface(bs, s, ll, target)
        if exact:
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=30,iota=iota0,G=G0)
            bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
            tf = ToroidalFlux(s, bs_tf, stellarator)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
        else:
#            import ipdb;ipdb.set_trace()
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual', hessian=False)
            bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
            tf = ToroidalFlux(s, bs_tf, stellarator)
            print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, ||residual||_2={np.linalg.norm(res['residual']):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
        
        boozer_surface_list.append(boozer_surface)
    return boozer_surface_list, coils, currents, ma, stellarator

#def compute_surfaces_in_NCSX(mpol=10, ntor=10, exact=True, Nt_coils=6, write_to_file = False):
#    DIR = os.path.dirname(os.path.realpath(__file__))
#    coils, currents, ma = get_ncsx_data(Nt_coils=Nt_coils, Nt_ma=10, ppp=20)
#    stellarator = CoilCollection(coils, currents, 3, True)
#
#    coils = stellarator.coils
#    currents = stellarator.currents
#    
#    stellsym = True
#    nfp = 3
#    
#    if exact:
#        nquadphi = ntor+1
#        nquadtheta = 2*mpol+1
#        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
#        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
#    else:
#        nquadphi = ntor + 1 + 5
#        nquadtheta = 2*mpol + 1 + 5
#        phis = np.linspace(0, 1/(2*nfp), ntor + 1 + 5, endpoint=False)
#        thetas = np.linspace(0, 1, 2*mpol + 1 + 5, endpoint=False)
#    
#    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
#    s.fit_to_curve(ma, 0.10, flip_theta=True)
#    iota0 = -0.4
#    
#    vol_list = np.linspace(-0.162, -3.3, 10)
#    G0 = 2. * np.pi * np.sum(np.abs(currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
#    boozer_surface_list = []
#    boozer_surface_dict = []
#
#    backup_dofs = s.get_dofs()
#    backup_iota = iota0
#    backup_G = G0
#
#    for idx,target in enumerate(vol_list):
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        label = Volume(s, stellarator)
#        boozer_surface = BoozerSurface(bs, s, label, target)
#        
#        try:
#            s.set_dofs(backup_dofs)
#            iota0 = backup_iota
#            G0 = backup_G
#
#            if exact:
#                res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=300,iota=iota0,G=G0)
#                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#                tf = ToroidalFlux(s, bs_tf, stellarator)
#                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}")
#            else:
#                res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-9, maxiter=30, constraint_weight=100., iota=iota0, G=G0, method='manual')
#                bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#                tf = ToroidalFlux(s, bs_tf, stellarator)
#                print(f"iter={res['iter']}, iota={res['iota']:.16f}, tf={tf.J():.16f}, vol={s.volume():.3f}, |label error|={np.abs(label.J()-target):.3e}, ||residual||_inf={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0), ord=np.inf):.3e}, ||grad||_inf = {np.linalg.norm(res['gradient'], ord=np.inf):.3e}")
#
#            if res['success']:
#                boozer_surface_list.append(boozer_surface)
#                boozer_surface_dict.append({'dofs': boozer_surface.surface.get_dofs(), 'iota': res['iota'], 'G': res['G']})
#                backup_dofs = s.get_dofs().copy()
#                backup_iota = res['iota']
#                backup_G = res['G']
#        except:
#            print("Didn't converge")
#    
#    with open(DIR + "/data/" + f"surface_dofs_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}.txt", 'w') as f:
#        for surf_dict in boozer_surface_dict:
#            np.savetxt(f, surf_dict['dofs'].reshape((1,-1)))
#    with open(DIR + '/data/' +        f"iotaG_mpol={mpol}_ntor={ntor}_nquadphi={nquadphi}_nquadtheta={nquadtheta}_stellsym={stellsym}_exact={exact}_Nt_coils={Nt_coils}.txt", 'w') as f:
#        for surf_dict in boozer_surface_dict:
#            f.write('%s\n' % (surf_dict['iota']))
#            f.write('%s\n' % (surf_dict['G']))
#    return boozer_surface_list, coils, currents, ma, stellarator

def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arclength-weight", type=float, default=0.)
    parser.add_argument("--curvature-weight", type=float, default=0.)
    parser.add_argument("--min-dist", type=float, default=0.)
    parser.add_argument("--dist-weight", type=float, default=0.)
    parser.add_argument("--Nt-coils", type=int, default=6)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["bfgs", "lbfgs", "newton-cg"])
    parser.add_argument("--runID", type=int, default=-1)
    args,_ = parser.parse_known_args()
    return args






    #import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection="3d")
    #for i in range(0, len(stellarator.coils)):
    #    ax = stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(stellarator._base_coils)])
    #ax.view_init(elev=90., azim=0)
    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-2, 2)
    #ax.set_zlim(-1, 1)
    #plt.show() 
    
    
    #def find_magnetic_axis(biotsavart, n, rguess, output='cylindrical'):
    #    assert output in ['cylindrical', 'cartesian']
    #    from scipy.spatial.distance import cdist
    #    from scipy.optimize import fsolve
    #    points = np.linspace(0, 2*np.pi, n, endpoint=False).reshape((n, 1))
    #    oneton = np.asarray(range(0, n)).reshape((n, 1))
    #    fak = 2*np.pi / (points[-1] - points[0] + (points[1]-points[0]))
    #    dists = fak * cdist(points, points, lambda a, b: a-b)
    #    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    #    if n % 2 == 0:
    #        D = 0.5 \
    #            * np.power(-1, cdist(oneton, -oneton)) \
    #            / np.tan(0.5 * dists)
    #    else:
    #        D = 0.5 \
    #            * np.power(-1, cdist(oneton, -oneton)) \
    #            / np.sin(0.5 * dists)
    #
    #    np.fill_diagonal(D, 0)
    #    D *= fak
    #    phi = points
    #
    #    def build_residual(rz):
    #        inshape = rz.shape
    #        rz = rz.reshape((2*n, 1))
    #        r = rz[:n ]
    #        z = rz[n:]
    #        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
    #        biotsavart.set_points(xyz)
    #        B = biotsavart.B()
    #        Bx = B[:, 0].reshape((n, 1))
    #        By = B[:, 1].reshape((n, 1))
    #        Bz = B[:, 2].reshape((n, 1))
    #        Br = np.cos(phi)*Bx + np.sin(phi)*By
    #        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
    #        residual_r = D @ r - r * Br / Bphi
    #        residual_z = D @ z - r * Bz / Bphi
    #        return np.vstack((residual_r, residual_z)).reshape(inshape)
    #
    #    def build_jacobian(rz):
    #        rz = rz.reshape((2*n, 1))
    #        r = rz[:n ]
    #        z = rz[n:]
    #        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
    #        biotsavart.set_points(xyz)
    #        GradB = biotsavart.dB_by_dX()
    #        B = biotsavart.B()
    #        Bx = B[:, 0].reshape((n, 1))
    #        By = B[:, 1].reshape((n, 1))
    #        Bz = B[:, 2].reshape((n, 1))
    #        dxBx = GradB[:, 0, 0].reshape((n, 1))
    #        dyBx = GradB[:, 1, 0].reshape((n, 1))
    #        dzBx = GradB[:, 2, 0].reshape((n, 1))
    #        dxBy = GradB[:, 0, 1].reshape((n, 1))
    #        dyBy = GradB[:, 1, 1].reshape((n, 1))
    #        dzBy = GradB[:, 2, 1].reshape((n, 1))
    #        dxBz = GradB[:, 0, 2].reshape((n, 1))
    #        dyBz = GradB[:, 1, 2].reshape((n, 1))
    #        dzBz = GradB[:, 2, 2].reshape((n, 1))
    #        cosphi = np.cos(phi)
    #        sinphi = np.sin(phi)
    #        Br = cosphi*Bx + sinphi*By
    #        Bphi = cosphi*By - sinphi*Bx
    #        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
    #        dzBr = cosphi*dzBx + sinphi*dzBy
    #        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
    #        dzBphi = cosphi*dzBy - sinphi*dzBx
    #        drBz = cosphi * dxBz + sinphi*dyBz
    #        # residual_r = D @ r - r * Br / Bphi
    #        # residual_z = D @ z - r * Bz / Bphi
    #        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
    #        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
    #        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
    #        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
    #        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])
    #    
    #    r0 = np.ones_like(phi) * rguess
    #    z0 = np.zeros_like(phi)
    #    x0 = np.vstack((r0, z0))
    #    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)
    #    if output == 'cylindrical':
    #        return np.hstack((soln[:n, None], phi, soln[n:, None]))
    #    else:
    #        return np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))

    #bs = BiotSavart(stellarator.coils, stellarator.currents)
    #sol = find_magnetic_axis(bs, 201, 1., output='cartesian')
    #Nt_ma = 25
    #nfp = 2
    #stellsym=True
    #ma = CurveRZFourier(np.linspace(0,1,201, endpoint=False), Nt_ma, nfp, stellsym)
    #ma.least_squares_fit(sol)
    #
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #g = ma.gamma()
    #ax.plot3D(g[:,0], g[:,1], g[:,2])
    #ax.plot3D(sol[:,0], sol[:,1], sol[:,2])

    #for i in range(0, len(stellarator.coils)):
    #    ax = stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(stellarator._base_coils)])
    #ax.view_init(elev=90., azim=0)
    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-2, 2)
    #ax.set_zlim(-1, 1)
    #plt.show()
    #print(np.linalg.norm(ma.gamma() - sol, ord=np.inf))
    #import ipdb;ipdb.set_trace() 
    





