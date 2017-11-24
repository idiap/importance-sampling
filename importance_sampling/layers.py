from keras import backend as K
from keras.engine import Layer
from keras import initializers


class BatchRenormalization(Layer):
    """Batch renormalization layer (Sergey Ioffe, 2017).
    
    # Arguments
        momentum: Momentum for the moving average
        epsilon: Added to variance to avoid divide by 0
        rmax: Maximum correction for the variance
        dmax: Maximum correction for the bias
    """
    def __init__(self, momentum=0.99, epsilon=1e-3, rmax_0=1., rmax_inf=3.,
                 dmax_0=0., dmax_inf=5., rmax_duration=40000,
                 dmax_duration=25000, **kwargs):
        super(BatchRenormalization, self).__init__(**kwargs)

        self.momentum = momentum
        self.epsilon = epsilon
        self.rmax_0 = rmax_0
        self.rmax_inf = rmax_inf
        self.rmax_dur = rmax_duration
        self.dmax_0 = dmax_0
        self.dmax_inf = dmax_inf
        self.dmax_dur = dmax_duration

    def build(self, input_shape):
        dim = input_shape[-1]
        if dim is None:
            raise ValueError(("The normalization axis should have a "
                              "defined dimension"))
        self.dim = dim

        # Trainable part
        self.gamma = self.add_weight(
            shape=(dim,),
            name="gamma",
            initializer=initializers.get("ones")
        )
        self.beta = self.add_weight(
            shape=(dim,),
            name="beta",
            initializer=initializers.get("zeros")
        )

        # Statistics
        self.moving_mean = self.add_weight(
            shape=(dim,),
            name="moving_mean",
            initializer=initializers.get("zeros"),
            trainable=False
        )
        self.moving_sigma = self.add_weight(
            shape=(dim,),
            name="moving_sigma",
            initializer=initializers.get("ones"),
            trainable=False
        )

        # rmax, dmax and steps
        self.steps = self.add_weight(
            shape=tuple(),
            name="steps",
            initializer=initializers.get("zeros"),
            trainable=False
        )
        self.rmax = self.add_weight(
            shape=tuple(),
            name="rmax",
            initializer=initializers.Constant(self.rmax_0),
            trainable=False
        )
        self.dmax = self.add_weight(
            shape=tuple(),
            name="dmax",
            initializer=initializers.Constant(self.dmax_0),
            trainable=False
        )

        self.built = True

    def _moments(self, x):
        axes = range(len(K.int_shape(x))-1)
        if K.backend() == "tensorflow":
            return K.tf.nn.moments(x, axes)
        else:
            # TODO: Maybe the following can be optimized a bit?
            mean = K.mean(K.reshape(x, (-1, self.dim)), axis=0)
            var = K.var(K.reshape(x, (-1, self.dim)), axis=0)

            return mean, var

    def _clip(self, x, x_min, x_max):
        if K.backend() == "tensorflow":
            return K.tf.clip_by_value(x, x_min, x_max)
        else:
            return K.maximum(K.minimum(x, x_max), x_min)

    def call(self, inputs, training=None):
        x = inputs
        assert not isinstance(x, list)

        # Compute the minibatch statistics
        mean, var = self._moments(x)
        sigma = K.sqrt(var + self.epsilon)

        # If in training phase set rmax, dmax large so that we use the moving
        # averages to do the normalization
        rmax = K.in_train_phase(self.rmax, K.constant(1e5), training)
        dmax = K.in_train_phase(self.dmax, K.constant(1e5), training)

        # Compute the corrections based on rmax, dmax
        r = K.stop_gradient(self._clip(
            sigma/self.moving_sigma,
            1./rmax,
            rmax
        ))
        d = K.stop_gradient(self._clip(
            (mean - self.moving_mean)/self.moving_sigma,
            -dmax,
            dmax
        ))

        # Actually do the normalization and the rescaling
        xnorm = ((x-mean)/sigma)*r + d
        y = self.gamma * xnorm + self.beta

        # Add the moving average updates
        self.add_update([
            K.moving_average_update(self.moving_mean, mean, self.momentum),
            K.moving_average_update(self.moving_sigma, sigma, self.momentum)
        ], x)

        # Add the r, d updates
        rmax_prog = K.minimum(1., self.steps/self.rmax_dur)
        dmax_prog = K.minimum(1., self.steps/self.dmax_dur)
        self.add_update([
            K.update_add(self.steps, 1),
            K.update(
                self.rmax,
                self.rmax_0 + rmax_prog*(self.rmax_inf-self.rmax_0)
            ),
            K.update(
                self.dmax,
                self.dmax_0 + dmax_prog*(self.dmax_inf-self.dmax_0)
            )
        ])

        return y
