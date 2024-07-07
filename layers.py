import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import ops

"Takes in Mel Spectrogram and sets random amount of random consecutive time frames to zero."
class RandomFrequencyMasking(layers.Layer):
    def __init__(self, F, nu, **kwargs):
        super(RandomFrequencyMasking, self).__init__(**kwargs)
        self.F = F
        self.nu = nu

    def call(self, inputs, training=False):
        if (training 
            # and self.rate>0
            ):
            f = tf.random.uniform([], minval=0, maxval=self.F, dtype=tf.int32)
            f0 = tf.random.uniform([], minval=0, maxval=self.nu - f, dtype=tf.int32)
            mask = tf.ones_like(inputs)
            mask = tf.concat([mask[:, :, :f0, :], tf.zeros_like(inputs[:, :, f0:f0 + f, :]), mask[:, :, f0 + f:, :]], axis=2)
            return inputs * mask

        # Apply the mask only during training
        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'F': self.F,
            'nu': self.nu,
        })
        return config

"Takes in Mel Spectrogram and sets random amount of random consecutive frequency frames to zero."
class RandomTimeMasking(layers.Layer):

    def __init__(self, T, tau, **kwargs):
        super(RandomTimeMasking, self).__init__(**kwargs)
        self.T = T
        self.tau = tau

    def call(self, inputs, training=False):
        if (training 
            # and self.rate>0
            ):
            t = tf.random.uniform([], minval=0, maxval=self.T, dtype=tf.int32)
            t0 = tf.random.uniform([], minval=0, maxval=self.tau - t, dtype=tf.int32)
            mask = tf.ones_like(inputs)
            mask = tf.concat([mask[:, :, :t0, :], tf.zeros_like(inputs[:, :, t0:t0 + t, :]), mask[:, :, t0 + t:, :]], axis=2)
            return inputs * mask

        # Apply the mask only during training
        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'tau': self.tau,
        })
        return config


    # def __init__(self, T, tau):
    #     super(RandomTimeMasking, self).__init__()
    #     self.T = T
    #     self.tau = tau
    
    # def call(self, inputs):
    #     t = random.randint(0, self.T)
    #     t0 = random.randrange(0, self.tau - t)
    #     mask = np.ones(inputs.shape)
    #     mask[:,t0:t0 + t] = 0
    #     return inputs * mask
