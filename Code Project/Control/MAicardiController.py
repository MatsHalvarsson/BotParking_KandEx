import tensorflow as tf

from .Controller import Controller




class MAicardiController(Controller):

    def __init__(self, epsilon, k:float=1.0, gamma:float=1.0, h:float=1.0):
        self.eps = epsilon
        self.k: tf.Tensor = tf.constant(k, dtype=tf.float32, shape=())
        self.g: tf.Tensor = tf.constant(gamma, dtype=tf.float32, shape=())
        self.h: tf.Tensor = tf.constant(h, dtype=tf.float32, shape=())
    
    def reinitialize(self):
        return self.reinitialize(keep_current_hyperparams=False)
    
    def reinitialize(self, k=1, gamma=1, h=1, keep_current_hyperparams=False):
        if keep_current_hyperparams:
            return
        self.k = tf.constant(k, dtype=tf.float32)
        self.g = tf.constant(gamma, dtype=tf.float32)
        self.h = tf.constant(h, dtype=tf.float32)

    def get_control_signal(self, s: tf.Tensor) -> tf.Tensor:
        u = self.g * tf.cos(s[1]) * s[0]
        w = self.k * s[1] + self.g * tf.cos(s[1]) * tf.sin(s[1]) * (1 + tf.sign(s[1]) * self.h * s[2] / (tf.abs(s[1]) + self.eps))
        return tf.convert_to_tensor((u, w), dtype=s.dtype)
    
    def train(self):
        print("Warning! Training of M. Aicardi controller is undefined. Returns super().train().")
        return super().train()