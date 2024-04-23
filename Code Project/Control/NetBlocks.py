import tensorflow as tf
from os.path import join as ospath_join




class DenseBlock(tf.Module):

    def __init__(self, n_neurons, input_dims, weights_initializer=None, bias_initializer=None, activation_fn=None, name=None):
        super().__init__(name=name)

        self.w_initializer:tf.initializers.Initializer = weights_initializer
        self.b_initializer:tf.initializers.Initializer = bias_initializer

        if not weights_initializer:        
            self.w_initializer = tf.initializers.GlorotNormal()
        if not bias_initializer:
            self.b_initializer = tf.initializers.Zeros()
        
        self.activation_fn = activation_fn
        if not activation_fn:
            self.activation_fn = tf.keras.activations.relu

        self.input_dims = input_dims
        self.output_dims = n_neurons
        self.dtype = tf.float32

        self._w: tf.Variable = tf.Variable(self.w_initializer((input_dims, n_neurons), self.dtype),
                                           trainable=True, dtype=self.dtype,
                                           shape=(input_dims, n_neurons),
                                           name="W")
        self._b: tf.Variable = tf.Variable(self.b_initializer((n_neurons,), self.dtype),
                                           trainable=True, dtype=self.dtype,
                                           shape=(n_neurons,),
                                           name="B")
        
        self.n_params = (input_dims + 1) * n_neurons
        
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.input_dims))
        return self.activation_fn(tf.matmul(x, self._w) + self._b)
    
    def call_shape_safe(self, x):
        return self.activation_fn(tf.matmul(x, self._w) + self._b)


class DenseNet(tf.Module):

    def __init__(self, layer_n_neurons:list[int], input_dims, name=None, **dense_kwargs):
        super().__init__(name=name)

        if len(layer_n_neurons) < 2:
            print("Warning! DenseNet was not built for a single layer only, or no layers, and might behave erroneously!")
        
        self.layers:list[DenseBlock] = []
        
        self.layers.append(DenseBlock(n_neurons=layer_n_neurons[0],
                                      input_dims=input_dims,
                                      **dense_kwargs))
        
        for units in layer_n_neurons[1:-1]:
            self.layers.append(DenseBlock(n_neurons=units,
                                          input_dims=self.layers[-1].output_dims,
                                          **dense_kwargs))
        
        dense_kwargs["activation_fn"] = tf.keras.activations.linear
        self.layers.append(DenseBlock(n_neurons=layer_n_neurons[-1],
                                      input_dims=self.layers[-1].output_dims,
                                      **dense_kwargs))

        self.input_dims = input_dims
        self.output_dims = layer_n_neurons[-1]
        self.var_cache:tuple[tf.Variable] = self.variables
        self.n_params = sum([net.n_params for net in self.layers])

    def __call__(self, x):
        for lyr in self.layers:
            x = lyr.call_shape_safe(x)
        return x
    
    def call_on_single_input(self, x):
        x = tf.reshape(x, (1, self.input_dims))
        for lyr in self.layers:
            x = lyr.call_shape_safe(x)
        x = tf.reshape(x, (self.output_dims,))
        return x
    
    def save_network(self, folder, name):
        for i, lyr in enumerate(self.layers):
            lyr._w.read_value().numpy().tofile(ospath_join(folder, name + "_lyr_" + str(i) + "_W" + ".csv"), sep=",")
            lyr._b.read_value().numpy().tofile(ospath_join(folder, name + "_lyr_" + str(i) + "_B" + ".csv"), sep=",")