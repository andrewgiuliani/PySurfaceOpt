import numpy as np
from jax import grad, vjp, jit
import jax.numpy as jnp
from simsopt._core.graph_optimizable import Optimizable


@jit
def curve_msc_pure(kappa, gammadash):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)

class MeanSquareCurvature(Optimizable):
    def __init__(self, curve, threshold):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[curve])
        self.curve = curve
        self.threshold = threshold
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))

    def msc(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    def J(self):
        return 0.5 * max(self.msc()-self.threshold, 0)**2

    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        deriv = self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)
        fak = max(self.msc()-self.threshold, 0.)
        return fak * deriv


