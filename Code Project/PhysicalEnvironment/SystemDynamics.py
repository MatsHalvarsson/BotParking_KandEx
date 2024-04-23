#import statements
from numpy import pi
from tensorflow import math as tfMath, Tensor as tfTensor, convert_to_tensor as asTensor, constant, float32, reshape


class SystemDynamics:

    """
    System dynamics model on the form 
        e' = - u * cos( a )
        a' = - w + u * sin( a ) / e
        d' = u * sin( a ) / e
    
    For numerical calculation on the derivatives, f(x)' = df(x) / dt => df(x) = RHS * dt, which in turn
    yields an incremental value for updating the system state variable, using, f(x_new) = f(x) + df(x).
    RHS meaning right hand side in the eq. system above.

    Class Methods:
        
        calculate_state_increments (e, a, d, u, w, dt):    |    polar rep vars e, a, d ; control inputs u, w ; time step length dt
            computes the increments of the polar state variables (e, a, d),
            uses the formula above.

    """

    def __init__(self, epsilon):
        self.eps = constant(epsilon, dtype=float32, shape=())

    #def calculate_state_increments(self, dt, e, a, d, u, w):
    def calculate_state_increments(self, dt: tfTensor, s: tfTensor, c: tfTensor) -> tfTensor:
        return asTensor((- c[0] * tfMath.cos(s[1]) * dt,
                         (- c[1] + c[0] * tfMath.sin(s[1]) / (s[0] + self.eps)) * dt,
                         c[0] * tfMath.sin(s[1]) / (s[0] + self.eps) * dt),
                         dtype=dt.dtype)


class NormalizedSystemDynamics(SystemDynamics):
    def __init__(self, epsilon):
        super().__init__(epsilon)
        self.norm_vector = constant([1.0, pi, pi], dtype=float32, shape=(3,))

    def calculate_state_increments(self, dt: tfTensor, s: tfTensor, c: tfTensor) -> tfTensor:
        return tfMath.divide(super().calculate_state_increments(dt, tfMath.multiply(self.norm_vector, s), c), self.norm_vector)