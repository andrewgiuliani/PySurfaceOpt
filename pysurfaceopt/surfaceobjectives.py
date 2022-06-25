import numpy as np
import scipy
from simsopt.geo.boozersurface import boozer_surface_residual
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt._core.graph_optimizable import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec

def forward_backward(P, L, U, rhs):
    """
    Solve a linear system of the form PLU*adj = rhs
    t(PLU)adj = rhs
    Ut * Lt * Pt adj = rhs
    
    Ut * y = rhs
    Lt * z = y
    Pt * adj = z
    """
    y = scipy.linalg.solve_triangular(U.T, rhs, lower=True) 
    z = scipy.linalg.solve_triangular(L.T, y, lower=False) 
    adj = P@z 
   
    #  iterative refinement
    yp = scipy.linalg.solve_triangular(U.T, rhs-(P@L@U).T@adj, lower=True) 
    zp = scipy.linalg.solve_triangular(L.T, yp, lower=False) 
    adj += P@zp 
 
    return adj



sDIM=25
class Volume:
    """
    Wrapper class for volume label.
    """

    def __init__(self, in_surface):
        #phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        sDIM = 10
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2

        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.volume()

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.d2volume_by_dcoeffdcoeff()
    
    def dJ_by_dcoils(self):
        return Derivative()

class Aspect_ratio:
    """
    Wrapper class for aspect ratio.
    """

    def __init__(self, in_surface):
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2

        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.aspect_ratio()

class Area:
    """
    Wrapper class for area.
    """

    def __init__(self, in_surface):
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2

        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.area()




class MajorRadius(Optimizable): 
    r"""
    This objective computes the major radius of a toroidal surface with the following formula:
    
    R_major = (1/2*pi) * (V / mean_cross_sectional_area)
    
    where mean_cross_sectional_area = \int^1_0 \int^1_0 z_\theta(xy_\varphi -yx_\varphi) - z_\varphi(xy_\theta-yx_\theta) ~d\theta d\varphi
    
    """
    def __init__(self, boozer_surface):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        #phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2


        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())
        
        self.volume = Volume(s)
        self.boozer_surface = boozer_surface
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = boozer_surface.bs
        self.recompute_bell()
    
    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None


    def compute(self, compute_derivatives=0):

        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
    
        g = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
        
        x = g[:, :, 0]
        y = g[:, :, 1]
        
        r = np.sqrt(x**2+y**2)
    
        xvarphi = g1[:, :, 0]
        yvarphi = g1[:, :, 1]
        zvarphi = g1[:, :, 2]
    
        xtheta = g2[:, :, 0]
        ytheta = g2[:, :, 1]
        ztheta = g2[:, :, 2]
    
        mean_area = np.mean((ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta))/r)/(2*np.pi)
        R_major = self.volume.J() / (2. * np.pi * mean_area)
        self._J = R_major
        
        bs = self.biotsavart
        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp if booz_surf.res['type'] == 'exact' else boozer_surface_dlsqgrad_dcoils_vjp
        
        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = self.dJ_dsurfacecoefficients()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
        self._dJ = -1 * adj_times_dg_dcoil

    def dJ_dsurfacecoefficients(self):
        """
        Return the objective value
        """
    
        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
    
        g = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
    
        dg_ds = surface.dgamma_by_dcoeff()
        dg1_ds = surface.dgammadash1_by_dcoeff()
        dg2_ds = surface.dgammadash2_by_dcoeff()
    
        x = g[:, :, 0, None]
        y = g[:, :, 1, None]
    
        dx_ds = dg_ds[:, :, 0, :]
        dy_ds = dg_ds[:, :, 1, :]
    
        r = np.sqrt(x**2+y**2)
        dr_ds = (x*dx_ds+y*dy_ds)/r
    
        xvarphi = g1[:, :, 0, None]
        yvarphi = g1[:, :, 1, None]
        zvarphi = g1[:, :, 2, None]
    
        xtheta = g2[:, :, 0, None]
        ytheta = g2[:, :, 1, None]
        ztheta = g2[:, :, 2, None]
    
        dxvarphi_ds = dg1_ds[:, :, 0, :]
        dyvarphi_ds = dg1_ds[:, :, 1, :]
        dzvarphi_ds = dg1_ds[:, :, 2, :]
    
        dxtheta_ds = dg2_ds[:, :, 0, :]
        dytheta_ds = dg2_ds[:, :, 1, :]
        dztheta_ds = dg2_ds[:, :, 2, :]
    
        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))
        dmean_area_ds = np.mean((1/(r**2))*((xvarphi * y * ztheta - xtheta * y * zvarphi + x * (-yvarphi * ztheta + ytheta * zvarphi)) * dr_ds + r * (-zvarphi * (ytheta * dx_ds - y * dxtheta_ds - xtheta * dy_ds + x * dytheta_ds) + ztheta * (yvarphi * dx_ds - y * dxvarphi_ds - xvarphi * dy_ds + x * dyvarphi_ds) + (-xvarphi * y + x * yvarphi) * dztheta_ds + (xtheta * y - x * ytheta) * dzvarphi_ds)), axis=(0, 1))
    
        dR_major_ds = (-self.volume.J() * dmean_area_ds + self.volume.dJ_by_dsurfacecoefficients() * mean_area) / mean_area**2
        return dR_major_ds



class ToroidalFlux(Optimizable):

    r"""
    Given a surface and Biot Savart kernel, this objective calculates

    .. math::
       J &= \int_{S_{\varphi}} \mathbf{B} \cdot \mathbf{n} ~ds, \\
       &= \int_{S_{\varphi}} \text{curl} \mathbf{A} \cdot \mathbf{n} ~ds, \\
       &= \int_{\partial S_{\varphi}} \mathbf{A} \cdot \mathbf{t}~dl,

    where :math:`S_{\varphi}` is a surface of constant :math:`\varphi`, and :math:`\mathbf A` 
    is the magnetic vector potential.
    """

    def __init__(self, in_boozer_surface, biotsavart, idx=0):
        Optimizable.__init__(self, depends_on=[in_boozer_surface, biotsavart])
        
        in_surface = in_boozer_surface.surface
        phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        self.surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        self.surface.set_dofs(in_surface.get_dofs())
        self.in_boozer_surface = in_boozer_surface

        self.biotsavart = biotsavart
        self.idx = idx
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        in_surface = self.in_boozer_surface.surface

        self.surface.set_dofs(in_surface.get_dofs())
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)
        
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        A = self.biotsavart.A()
        self._J = np.sum(A * xtheta)/ntheta
        
        booz_surf = self.in_boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp if booz_surf.res['type'] == 'exact' else boozer_surface_dlsqgrad_dcoils_vjp
        
        dJ_by_dA = self.dJ_by_dA()
        dJ_by_dcoils = self.biotsavart.A_vjp(dJ_by_dA)
        
        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = self.dJ_by_dsurfacecoefficients()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dA(self):
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        dJ_by_dA = np.zeros((ntheta, 3))
        dJ_by_dA[:, ] = xtheta/ntheta
        return dJ_by_dA

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dA_by_dX = self.biotsavart.dA_by_dX()
        A = self.biotsavart.A()
        dgammadash2 = self.surface.gammadash2()[self.idx, :]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx, :]

        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        term1 = np.sum(dA_dc * dgammadash2[..., None], axis=(0, 1))
        term2 = np.sum(A[..., None] * dgammadash2_by_dc, axis=(0, 1))

        out = (term1+term2)/ntheta
        return out

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta, 3, 3, 3))
        dA_by_dX = self.biotsavart.dA_by_dX()
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc, optimize=True)

        dgammadash2 = self.surface.gammadash2()[self.idx]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]

        term1 = np.sum(d2A_dcdc * dgammadash2[..., None, None], axis=-3)
        term2 = np.sum(dA_dc[..., :, None] * dgammadash2_by_dc[..., None, :], axis=-3)
        term3 = np.sum(dA_dc[..., None, :] * dgammadash2_by_dc[..., :, None], axis=-3)

        out = (1/ntheta) * np.sum(term1+term2+term3, axis=0)
        return out

class BoozerResidual(Optimizable):
    r"""
    """
    def __init__(self, boozer_surface, bs, constraint_weight):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface=boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        # same number of points as on the solved surface
        #phis   = in_surface.quadpoints_phi
        #thetas = in_surface.quadpoints_theta
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        phis+=phis[0]/2.

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.constraint_weight=constraint_weight
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1,3)))
 
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        if self.boozer_surface.res['type'] == 'exact':
            w = None
        else:
            w = self.boozer_surface.res['weighting']

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1,weighting=w)
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)] ) )
        self._J = 0.5*np.sum(rtil**2)
        
        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp if booz_surf.res['type'] == 'exact' else boozer_surface_dlsqgrad_dcoils_vjp

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dl[:-2] = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil
        
        if booz_surf.res['type'] == 'lscons':
            if booz_surf.surface.stellsym:
                dJ_ds = np.concatenate((dJ_ds, [0.]))
            else:
                dJ_ds = np.concatenate((dJ_ds, [0., 0.]))

        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
        self._dJ = dJ_by_dcoils -  adj_times_dg_dcoil
        
    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        
        surface = self.surface
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        if self.boozer_surface.res['type'] == 'exact':
            w = None
        else:
            w = self.boozer_surface.res['weighting']

        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weighting=w)

        r/=np.sqrt(num_points)
        r_dB/=np.sqrt(num_points)
        
        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB




class NonQuasiAxisymmetricRatio(Optimizable):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasiaxisymmetric and
    non-quasiaxisymmetric components, then returns a penalty on the latter component.
    
    .. math::
        J &= \frac{1}{2}\int_{\Gamma_{s}} (B-B_{\text{QS}})^2~dS
          &= \frac{1}{2}\int_0^1 \int_0^1 (B - B_{\text{QS}})^2 \|\mathbf n\| ~d\varphi~\d\theta 
    where
    
    .. math::
        B &= \| \mathbf B(\varphi,\theta) \|_2
        B_{\text{QS}} &= \frac{\int_0^1 \int_0^1 B \| n\| ~d\varphi ~d\theta}{\int_0^1 \int_0^1 \|\mathbf n\| ~d\varphi ~d\theta}
    
    """
    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[boozer_surface])
        in_surface=boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1,3)))
        
        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        
        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_nonQS = modB - B_QS[None, :]
        self._J = np.mean(dS * B_nonQS**2)  / np.mean(dS * B_QS**2)

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp if booz_surf.res['type'] == 'exact' else boozer_surface_dlsqgrad_dcoils_vjp

        dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)
        
        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.concatenate((self.dJ_by_dsurfacecoefficients(), [0., 0.]))
        
        if booz_surf.res['type'] == 'lscons':
            if booz_surf.surface.stellsym:
                dJ_ds = np.concatenate((dJ_ds, [0.]))
            else:
                dJ_ds = np.concatenate((dJ_ds, [0., 0.]))

        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
        self._dJ = dJ_by_dcoils-adj_times_dg_dcoil


    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        denom = np.mean(dS, axis=0)
        B_QS = np.mean(modB * dS, axis=0) / denom
        B_nonQS = modB - B_QS[None, :]
        
        dmodB_dB = B / modB[..., None]
        dnum_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta) # d J_nonQS / dB_ijk
        ddenom_by_dB = B_QS[None,:, None] * dmodB_dB * dS[:,:,None]  / (nphi*ntheta) # dJ_QS/dB_ijk
        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        return (denom * dnum_by_dB - num * ddenom_by_dB) / denom**2 
    
    def dJ_by_dsurfacecoefficients(self):
        """
        Return the partial derivative of the objective with respect to the surface coefficients
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_nonQS = modB - B_QS[None, :]
        
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
        
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]
        
        num = np.mean(modB * dS, axis=0)
        denom = np.mean(dS, axis=0)
        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=0) 
        ddenom_dc = np.mean(dS_dc, axis=0)
        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2
        B_nonQS_dc = dmodB_dc - B_QS_dc[None, :]
        
        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        dnum_by_dc = np.mean(0.5*dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1)) 
        ddenom_by_dc = np.mean(0.5*dS_dc * B_QS[..., None]**2 + dS[..., None] * B_QS[..., None] * B_QS_dc, axis=(0, 1)) 
        dJ_by_dc = (denom * dnum_by_dc - num * ddenom_by_dc ) / denom**2 
        return dJ_by_dc

class Iotas(Optimizable):
    def __init__(self, boozer_surface):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.bs 
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None
    
    def compute(self):
        self._J = self.boozer_surface.res['iota']
        
        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp if booz_surf.res['type'] == 'exact' else boozer_surface_dlsqgrad_dcoils_vjp

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        if booz_surf.res['type'] == 'lscons':
            if booz_surf.surface.stellsym:
                dJ_ds[-3] = 1
                #dJ_ds = np.concatenate((dJ_ds, [0.]))
            else:
                dJ_ds[-4] = 1
                #dJ_ds = np.concatenate((dJ_ds, [0., 0.]))
        else:
            dJ_ds[-2] = 1.


        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
        self._dJ = -1.*adj_times_dg_dcoil
        
def boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dresidual_dcoils:
    lm^T dresidual_dcoils    = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcoils
    lm^T dresidual_dcurrents = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcurrents
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """
    surface = booz_surf.surface
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    res, dres_dB = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0)
    dres_dB = dres_dB.reshape((-1, 3, 3))

    lm_label = lm[-1]
    lmask = np.zeros(booz_surf.res["mask"].shape)
    lmask[booz_surf.res["mask"]] = lm[:-1]
    lm_cons = lmask.reshape((-1, 3))
   
    lm_times_dres_dB = np.sum(lm_cons[:, :, None] * dres_dB, axis=1).reshape((-1, 3))
    lm_times_dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    lm_times_dlabel_dcoils = lm_label*booz_surf.label.dJ_by_dcoils()
    return lm_times_dres_dcoils+lm_times_dlabel_dcoils


def boozer_surface_dlsqgrad_dcoils_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dlsqgrad_dcoils, \lm^T * dlsqgrad_dcurrents:
    lm^T dresidual_dcoils    = lm^T [dr_dsurface]^T[dr_dcoils]    + sum r_i lm^T d2ri_dsdc
    lm^T dresidual_dcurrents = lm^T [dr_dsurface]^T[dr_dcurrents] + sum r_i lm^T d2ri_dsdcurrents
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """
    
    if booz_surf.res['type'] == 'lscons':
        if booz_surf.surface.stellsym:
            lm = lm[:-1]
        else:
            lm = lm[:-2]

    #lm_label = lm[-1]
    surface = booz_surf.surface
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    num_points = 3 * nphi * ntheta
    # r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    boozer = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=1, weighting=booz_surf.res['weighting'])
    r = boozer[0]/np.sqrt(num_points)
    dr_dB = boozer[1].reshape((-1, 3, 3))/np.sqrt(num_points)
    dr_ds = boozer[2]/np.sqrt(num_points)
    d2r_dsdB = boozer[3]/np.sqrt(num_points)
    d2r_dsdgradB = boozer[4]/np.sqrt(num_points)

    v1 = np.sum(np.sum(lm[:, None]*dr_ds.T, axis=0).reshape((-1, 3, 1)) * dr_dB, axis=1)
    v2 = np.sum(r.reshape((-1, 3, 1))*np.sum(lm[None, None, :]*d2r_dsdB, axis=-1).reshape((-1, 3, 3)), axis=1)
    v3 = np.sum(r.reshape((-1, 3, 1, 1))*np.sum(lm[None, None, None, :]*d2r_dsdgradB, axis=-1).reshape((-1, 3, 3, 3)), axis=1)
    dres_dcoils = biotsavart.B_and_dB_vjp(v1+v2, v3)
    return dres_dcoils[0]+dres_dcoils[1]
    #dres_dcoils = [a + b for a, b in zip(dres_dcoils[0], dres_dcoils[1])]

    #lm_times_dres_dB = v1 + v2
    #lm_times_dres_dgradB = v3
    #dB_by_dcoilcurrents = biotsavart.dB_by_dcoilcurrents()
    #d2B_by_dXdcoilcurrents = biotsavart.d2B_by_dXdcoilcurrents()
    #dres_dcurrents = [np.sum(lm_times_dres_dB*dB_dcurr) + np.sum(lm_times_dres_dgradB * dgradB_dcurr) for dB_dcurr, dgradB_dcurr in zip(dB_by_dcoilcurrents, d2B_by_dXdcoilcurrents)]

    #return dres_dcoils, dres_dcurrents



def boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0, weighting=None):
    """
    For a given surface with points x on it, this function computes the
    differentiated residual
       d/dB[ G*B_BS(x) - ||B_BS(x)||^2 * (x_phi + iota * x_theta) ]
    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """
    
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang

    GI = np.eye(3, 3) * G
    dresidual_dB = GI[None, None, :, :] - 2. * tang[:, :, :, None] * B[:, :, None, :]

    if weighting == '1/B':
        B2 = np.sum(B**2, axis=2)
        modB = np.sqrt(B2)
        w = 1./modB
        dw_dB = -B/B2[:, :, None]**1.5



    if weighting is not None:
        rtil = w[:, :, None] * residual
        drtil_dB = residual[:, :, :, None] * dw_dB[:, :, None, :] + dresidual_dB * w[:, :, None, None] 
    else:
        rtil = residual.copy()
        drtil_dB = dresidual_dB.copy()

    rtil_flattened = rtil.reshape((nphi*ntheta*3, ))
    drtil_dB_flattened = drtil_dB.reshape((nphi*ntheta*3, 3))

    if derivatives == 0:
        return rtil_flattened, drtil_dB_flattened

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
    dtang_dc = dxphi_dc + iota * dxtheta_dc
    dresidual_dc = G*dB_dc \
        - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - np.sum(B**2, axis=2)[..., None, None] * dtang_dc
    dresidual_diota = -np.sum(B**2, axis=2)[..., None] * xtheta

    d2residual_dcdB = -2*dB_dc[:, :, None, :, :] * tang[:, :, :, None, None] - 2*B[:, :, None, :, None] * dtang_dc[:, :, :, None, :]
    d2residual_diotadB = -2.*B[:, :, None, :] * xtheta[:, :, :, None]
    d2residual_dcdgradB = -2.*B[:, :, None, None, :, None]*dx_dc[:, :, None, :, None, :]*tang[:, :, :, None, None, None]
    idx = np.arange(3)
    d2residual_dcdgradB[:, :, idx, :, idx, :] += dx_dc * G

    if weighting == '1/B':
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/modB[:, :, None]
        dw_dc =  -dmodB_dc/B2[:, :, None]
        
        d2w_dcdB = -(dB_dc * B2[:, :, None, None]**1.5 - 1.5*dB2_dc[:, :, None, :]*modB[:, :, None, None]*B[:, :, :, None])/B2[:, :, None, None]**3
        d2w_dcdgradB = dw_dB[:, :, None, :, None] * dx_dc[:, :, :, None, :]

    if weighting is not None:
        drtil_dc        = dresidual_dc * w[:, :, None, None] + dw_dc[:, :, None, :] * residual[..., None]
        drtil_diota     = w[:, :, None] * dresidual_diota
        d2rtil_dcdB     = dresidual_dc[:, :, :, None, :]*dw_dB[:, :, None, :, None]  \
                        + dresidual_dB[:, :, :, :, None]*dw_dc[:, :, None, None, :] \
                        + d2residual_dcdB*w[:, :, None, None, None] \
                        + residual[:, :, :, None, None]*d2w_dcdB[:, :, None, :, :]
        d2rtil_diotadB  = dw_dB[:, :, None, :]*dresidual_diota[:, :, :, None] + w[:, :, None, None]*d2residual_diotadB 
        d2rtil_dcdgradB = d2w_dcdgradB[:, :, None, :, :, :]*residual[:, :, :, None, None, None] + d2residual_dcdgradB*w[:, :, None, None, None, None]
    else:
        drtil_dc        = dresidual_dc.copy()
        drtil_diota     = dresidual_diota.copy()
        d2rtil_dcdB     = d2residual_dcdB.copy()
        d2rtil_diotadB  = d2residual_diotadB.copy()
        d2rtil_dcdgradB = d2residual_dcdgradB.copy()


    drtil_dc_flattened = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))
    d2rtil_dcdB_flattened = d2rtil_dcdB.reshape((nphi*ntheta*3, 3, nsurfdofs))
    d2rtil_diotadB_flattened = d2rtil_diotadB.reshape((nphi*ntheta*3, 3, 1))
    d2rtil_dcdgradB_flattened = d2rtil_dcdgradB.reshape((nphi*ntheta*3, 3, 3, nsurfdofs))
    d2rtil_diotadgradB_flattened = np.zeros((nphi*ntheta*3, 3, 3, 1))

    if user_provided_G:
        dresidual_dG = B
        d2residual_dGdB = np.ones((nphi*ntheta, 3, 3))
        d2residual_dGdB[:, :, :] = np.eye(3)[None, :, :]
        d2residual_dGdB = d2residual_dGdB.reshape((nphi,ntheta, 3, 3))
        d2residual_dGdgradB = np.zeros((nphi, ntheta, 3, 3, 3))

        if weighting is not None:
            drtil_dG = dresidual_dG * w[:, :, None]
            d2rtil_dGdB = d2residual_dGdB * w[:, :, None, None] + dw_dB[:, :, None, :]*dresidual_dG[:, :, :, None]
            d2rtil_dGdgradB = d2residual_dGdgradB.copy()
        else:
            drtil_dG = dresidual_dG.copy()
            d2rtil_dGdB = d2residual_dGdB.copy()
            d2rtil_dGdgradB = d2residual_dGdgradB.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        d2rtil_dGdB_flattened = d2rtil_dGdB.reshape((nphi*ntheta*3, 3, 1))
        d2rtil_dGdgradB_flattened = d2rtil_dGdgradB.reshape((nphi*ntheta*3, 3, 3, 1))

        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
        d2rtil_dsurfacedB = np.concatenate((d2rtil_dcdB_flattened,
                                            d2rtil_diotadB_flattened,
                                            d2rtil_dGdB_flattened), axis=-1)
        d2rtil_dsurfacedgradB = np.concatenate((d2rtil_dcdgradB_flattened,
                                                d2rtil_diotadgradB_flattened,
                                                d2rtil_dGdgradB_flattened), axis=-1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)
        d2rtil_dsurfacedB = np.concatenate((d2rtil_dcdB_flattened, d2rtil_diotadB_flattened), axis=-1)
        d2rtil_dsurfacedgradB = np.concatenate((d2rtil_dcdgradB_flattened, d2rtil_diotadgradB_flattened), axis=-1)

    if derivatives == 1:
        return rtil_flattened, drtil_dB_flattened, J, d2rtil_dsurfacedB, d2rtil_dsurfacedgradB
