import numpy as np
import tensorflow as tf

from .StateInitializers import StateInitializer



class SystemState:

    """
    Representation alternatives:
        
        Polar rep:
            e := distance of vehicle-to-goal vector (positional error)
            a := 'alfa'  | angle of vehicle orientation to vehicle-to-goal direction (alignment error)
            d := 'theta' | angle of vehicle-to-goal vector from x-axis
        
        Cartesian rep:
            x := ..
            y := ..
            p := 'phi' | vehicle rotation from x-axis
    
    Class variables:

        __state := underlying memory for (e, a, d) :: np.ndarray
    
    Class properties:
        
        e := e | readonly
        a := a | readonly
        d := d | readonly
        polar_rep := (e, a, d) | readonly
        cartesian_rep := (x, y, p) | readonly

    Class operations:

        __init__ :
            ...
        
        initialize_state (polar_state_values):
            sets the (e,a,d) state variables to values in argument
        
        update_state_variables :
            Used for updating the internal variables of the environment state.

        get_cartesian_rep :
            transform from internal variables to cartesian representation and return result.
        
    """

    def __init__(self, state_initializer:StateInitializer, value_dtype=tf.float32):
        self.s_initer = state_initializer
        self._s = tf.Variable(state_initializer.s_init(), dtype=value_dtype, shape=(3,), trainable=False)

    @property
    def current_state(self) -> tf.Tensor:
        return self._s.read_value()
    
    @property
    def current_state_cartesian(self):
        return self.convert_to_cartesian(self._s)
    
    def initialize_state(self):
        self._s.assign(self.s_initer.s_init(), read_value=False)

    def update_state_variables(self, incs):
        e_raw = self._s[0] + incs[0]
        a = self._s[1] + incs[1]
        if e_raw < 0:
            a += np.pi
        a = tf.math.atan2(tf.math.sin(a), tf.math.cos(a))
        #tf.assert_less(a, 3*np.pi, "Warning! Huge update to alfa!")
        #tf.assert_greater(a, -2*np.pi, "Warning! Huge update to alfa!")
        #if a >= np.pi:
        #    a -= 2*np.pi
        #if a < -np.pi:
        #    a += 2*np.pi
        d = self._s[2] + incs[2]
        if e_raw < 0:
            d += np.pi
        d = tf.math.atan2(tf.math.sin(d), tf.math.cos(d))
        #tf.assert_less(d, 3*np.pi, "Warning! Huge update to theta!")
        #tf.assert_greater(d, -2*np.pi, "Warning! Huge update to theta!")
        #if d >= np.pi:
        #    d -= 2*np.pi
        #if d < -np.pi:
        #    d += 2*np.pi
        self._s.assign((tf.abs(e_raw), a, d), read_value=False)
    
    @staticmethod
    def convert_to_cartesian(polar_states: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor([(- s[0] * tf.cos(s[2]),
                                      - s[0] * tf.sin(s[2]),
                                      tf.math.atan2(tf.math.sin(s[2] - s[1]), tf.math.cos(s[2] - s[1])))
                                     for s in polar_states])


class NormalizedSystemState(SystemState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_vector = tf.constant([1.0, np.pi, np.pi], dtype=tf.float32, shape=(3,))
    
    def update_state_variables(self, incs):
        e_raw = self._s[0] + incs[0]
        a = self._s[1] + incs[1]
        if e_raw < 0:
            a += 1.0
        #tf.assert_less(a, 3.0, "Warning! Huge update to alfa!")
        #tf.assert_greater(a, -2.0, "Warning! Huge update to alfa!")
        if a >= 1.0:
            a -= 2.0
        if a < -1.0:
            a += 2.0
        d = self._s[2] + incs[2]
        #tf.assert_less(d, 3.0, "Warning! Huge update to theta!")
        #tf.assert_greater(d, -2.0, "Warning! Huge update to theta!")
        if d >= 1.0:
            d -= 2.0
        if d < -1.0:
            d += 2.0
        self._s.assign((tf.abs(e_raw), a, d), read_value=False)
    
    @staticmethod
    def convert_to_cartesian(polar_states: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor([(- s[0] * tf.cos(s[2] * np.pi),
                                      - s[0] * tf.sin(s[2] * np.pi),
                                      tf.math.atan2(tf.math.sin((s[2] - s[1])*np.pi), tf.math.cos((s[2] - s[1])*np.pi)))
                                     for s in polar_states])