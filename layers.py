import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import ops
#from tensorflow.keras.src.api_export import keras_export

#Takes in Mel Spectrogram and sets random amount of random consecutive time frames to zero.
#@keras_export('keras.layers.experimental.RandomFrequencyMasking')
class RandomFrequencyMasking(layers.Layer):
    def __init__(self, F, nu, seed=None, **kwargs):
        super(RandomFrequencyMasking, self).__init__(**kwargs)
        self.F = F
        self.nu = nu
        self.seed = seed

    def call(self, inputs, training=False):
        if (training 
            # and self.rate>0
            ):
            tf.random.set_seed(self.seed)
            f = tf.random.uniform([], minval=0, maxval=self.F, dtype=tf.int32, seed=self.seed)
            f0 = tf.random.uniform([], minval=0, maxval=self.nu - f, dtype=tf.int32, seed=self.seed)
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
            'seed': self.seed,
        })
        return config

#Takes in Mel Spectrogram and sets random amount of random consecutive frequency frames to zero.
#@keras_export('keras.layers.experimental.RandomTimeMasking')
class RandomTimeMasking(layers.Layer):

    def __init__(self, T, tau, seed=None, **kwargs):
        super(RandomTimeMasking, self).__init__(**kwargs)
        self.T = T
        self.tau = tau
        self.seed = seed

    def call(self, inputs, training=False):
        if (training 
            # and self.rate>0
            ):
            tf.random.set_seed(self.seed)
            t = tf.random.uniform([], minval=0, maxval=self.T, dtype=tf.int32, seed=self.seed)
            t0 = tf.random.uniform([], minval=0, maxval=self.tau - t, dtype=tf.int32, seed=self.seed)
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
            'seed': self.seed,
        })
        return config
