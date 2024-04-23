from tensorflow import random, constant, float32, abs as tfAbs, multiply, reduce_mean




class RandGenHolder:
    gen:random.Generator = random.Generator(state=random.create_rng_state(seed=69000, alg='philox'), alg='philox')

    @staticmethod
    def set_generator_seed(seed=69000):
        RandGenHolder.gen.reset_from_seed(seed)

def exploration_with_probability(prob, expl_fn):
    gen = RandGenHolder.gen
    prob = constant(prob, dtype=float32, shape=())
    def exploration(*args, **kwargs):
        if gen.uniform(shape=(), minval=0.0, maxval=1.0) < prob:
            return expl_fn(*args, **kwargs)
        return no_exploration(*args, **kwargs)
    return exploration

def no_exploration(c):
    return c

def gaussian_addition(c):
    return c + RandGenHolder.gen.normal(c.shape, stddev=reduce_mean(tfAbs(c)))

def gauss_add_withstddev(stddev):
    def gauss_add(c):
        return c + RandGenHolder.gen.normal(c.shape, stddev=stddev)
    return gauss_add

def gauss_add_relative_stddev(fraction):
    def gauss_add(c):
        return c + RandGenHolder.gen.normal(c.shape, stddev=multiply(reduce_mean(tfAbs(c)), fraction))
    return gauss_add