from tensorflow import random, float32 as tfFloat32, constant, concat
from numpy import pi as pi



class StateInitializer:
    def s0_init(self):
        return 0
    
    def s1_init(self):
        return 0
    
    def s2_init(self):
        return 0
    
    def s_init(self):
        return (0,0,0)
    
    def __str__(self):
        return str(type(self))

class AlfaStateInitializer(StateInitializer):

    def __init__(self, e_min=5.0, e_max=20.0, a_min=-pi, a_max=pi, d_min=-pi, d_max=pi, dtype=tfFloat32,
                 seed=69):
        self.e_min = constant(e_min, dtype=dtype, shape=())
        self.e_max = constant(e_max, dtype=dtype, shape=())
        self.a_min = constant(a_min, dtype=dtype, shape=())
        self.a_max = constant(a_max, dtype=dtype, shape=())
        self.d_min = constant(d_min, dtype=dtype, shape=())
        self.d_max = constant(d_max, dtype=dtype, shape=())
        self.dtype = dtype
        self.gen = random.Generator(state=random.create_rng_state(1337*seed, alg='philox'), alg='philox')

    def s0_init(self):
        return self.gen.uniform(shape=(), minval=self.e_min, maxval=self.e_max, dtype=self.dtype)
    
    def s1_init(self):
        return self.gen.uniform(shape=(), minval=self.a_min, maxval=self.a_max, dtype=self.dtype)
    
    def s2_init(self):
        return self.gen.uniform(shape=(), minval=self.d_min, maxval=self.d_max, dtype=self.dtype)
    
    def s_init(self):
        return self.gen.uniform(shape=(3,), minval=(self.e_min, self.a_min, self.d_min),
                                maxval=(self.e_max, self.a_max, self.d_max), dtype=self.dtype)

class NormalizedRandomInitializer(StateInitializer):

    def __init__(self, seed=69):
        #self.normvalue = constant(1.0, dtype=tfFloat32, shape=())
        self.gen = random.Generator(state=random.create_rng_state(1337*seed, alg='philox'), alg='philox')
    
    def s_init(self):
        return concat([[1.0], self.gen.uniform(shape=(2,), minval=-1.0, maxval=1.0, dtype=tfFloat32)], axis=0)

class ConstantInitializer(StateInitializer):

    def __init__(self, state_0):
        self.s = constant(state_0, dtype=tfFloat32)
        self.s_str = str(state_0)
    
    def s_init(self):
        return self.s
    
    def __str__(self):
        return "ConstantInitializer: " + self.s_str