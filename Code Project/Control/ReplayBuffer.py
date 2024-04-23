from tensorflow import TensorArray as TA, Tensor, Variable, reshape, concat, constant, range as tfRange, round as tfRound, cast
from tensorflow import float32 as tf_f32, int32 as tf_int
from tensorflow import random




class ReplayBuffer:

    def __init__(self, num_expected_transitions=1000, expected_batch_size=16,
                 num_state_vars=3, num_control_vars=2):
        self.n_ex_tr = num_expected_transitions
        self.ex_bs = constant(expected_batch_size, dtype=tf_int, shape=())
        self.n_svars = num_state_vars
        self.n_cvars = num_control_vars
        self.reset()
    
    def __str__(self):
        return "Shuffling once and traversing all transitions replay buffer"
    
    def reset(self, expected_batch_size:None | int=None, keep_fraction=0.0):
        if expected_batch_size is not None:
            self.ex_bs = constant(expected_batch_size, dtype=tf_int)
        
        if hasattr(self, '_ta'):
            if self._ta is not None:
                old_ta = self._ta
                if self._n_tr < 20:
                    keep_fraction = 0.0

                if keep_fraction == 0.0:
                    _ = old_ta.close()
                else:
                    keep_count = cast(tfRound(cast(self._n_tr, tf_f32) * keep_fraction), tf_int)
                    keep_idxs = random.shuffle(tfRange(0, self._n_tr, dtype=tf_int))[0 : keep_count]
                    self._ta = TA(dtype=tf_f32, size=self.n_ex_tr + keep_count, dynamic_size=True,
                                  clear_after_read=False, tensor_array_name="replay_buffer",
                                  element_shape=(2 * self.n_svars + self.n_cvars + 1 + 1,))
                    self._ta.scatter(tfRange(0,keep_count, dtype=tf_int), old_ta.gather(keep_idxs))
                    self._n_tr = constant(keep_count, dtype=tf_int)
                    self._batch_idxs_shuffle = random.shuffle(tfRange(0, self._n_tr, dtype=tf_int))
                    self._shuffler_size = constant(self._n_tr, dtype=tf_int)
                    self._i_shuffle = constant(0, dtype=tf_int)
                    _ = old_ta.close()
                    return
            
        self._ta = TA(dtype=tf_f32, size=self.n_ex_tr, dynamic_size=True,
                      clear_after_read=False, tensor_array_name="replay_buffer",
                      element_shape=(2 * self.n_svars + self.n_cvars + 1 + 1,))
        self._n_tr = constant(0, dtype=tf_int)
        self._batch_idxs_shuffle = tfRange(0, self.ex_bs, dtype=tf_int)
        self._shuffler_size = constant(0, dtype=tf_int)
        self._i_shuffle = constant(0, dtype=tf_int)
    
    def add_transition(self, old_s, a, r, new_s, goal):
        self._ta = self._ta.write(self._n_tr, concat((old_s, a, reshape(r, -1), new_s, reshape(goal, -1)), axis=0))
        self._n_tr += 1
    
    def sample_minibatch(self, batch_size: Tensor):
        if self.ex_bs != batch_size:
            print("Warning! Batch size changed since last replay buffer reset! Implementation does not handle that currently.")
            if self._n_tr <= batch_size:
                self.ex_bs = batch_size
                self._batch_idxs_shuffle = tfRange(0, batch_size, dtype=tf_int)
        
        if self._n_tr <= batch_size:
            self._i_shuffle = self._n_tr
            return self._ta.gather(self._batch_idxs_shuffle[:self._n_tr])
        
        if self._i_shuffle + batch_size > self._n_tr:
            self._batch_idxs_shuffle = random.shuffle(tfRange(0, self._n_tr, dtype=tf_int))
            self._shuffler_size = self._n_tr
            idxs = self._batch_idxs_shuffle[0 : batch_size]
            self._i_shuffle = batch_size
        
        elif self._i_shuffle > self._shuffler_size:
            idxs = tfRange(self._i_shuffle, self._i_shuffle + batch_size, dtype=tf_int)
            self._i_shuffle += batch_size
        
        elif self._i_shuffle + batch_size > self._shuffler_size:
            idxs = concat((self._batch_idxs_shuffle[self._i_shuffle:], tfRange(self._shuffler_size, self._i_shuffle + batch_size, dtype=tf_int)), axis=0)
            self._i_shuffle += batch_size
        
        else:
            idxs = self._batch_idxs_shuffle[self._i_shuffle:self._i_shuffle + batch_size]
            self._i_shuffle += batch_size
        
        return self._ta.gather(idxs)

class AllRandomBuffer(ReplayBuffer):
    def __init__(self, seed=69*69, **superkwargs):
        self._rand_gen = random.Generator(state=random.create_rng_state(seed=seed, alg='philox'), alg='philox')
        super().__init__(**superkwargs)
    
    def __str__(self):
        return "Uniform Random sampling replay buffer"
    
    def sample_minibatch(self, batch_size: Tensor, include_n_newest_transitions=0):
        n = include_n_newest_transitions
        if self.ex_bs != batch_size:
            print("Warning! Batch size changed since last replay buffer reset!")
            self.ex_bs = batch_size
        
        if self._n_tr <= batch_size:
            return self._ta.stack()
        
        idxs = self._rand_gen.uniform(shape=(batch_size - n,), minval=0, maxval=self._n_tr - n, dtype=tf_int)
        if n > 0:
            idxs = concat([idxs, tfRange(self._n_tr-n, self._n_tr, dtype=tf_int)], axis=0)
        
        return self._ta.gather(idxs)