import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras import ops

"Takes in Mel Spectrogram and sets random amount of random consecutive time frames to zero."
class RandomFrequencyMasking(layers.Layer):
    def __init__(self, F, nu):
        super(RandomFrequencyMasking, self).__init__()
        self.F = F
        self.nu = nu

    def call(self, inputs):
        f = random.randint(0, self.F)
        f0 = random.randrange(0, self.nu - f)
        mask = np.ones(inputs.shape)
        mask[f0:f0 + f,:] = 0
        return inputs * mask

"Takes in Mel Spectrogram and sets random amount of random consecutive frequency frames to zero."
class RandomTimeMasking(layers.Layer):
    def __init__(self, T, tau):
        super(RandomTimeMasking, self).__init__()
        self.T = T
        self.tau = tau
    
    def call(self, inputs):
        t = random.randint(0, self.T)
        t0 = random.randrange(0, self.tau - t)
        mask = np.ones(inputs.shape)
        mask[:,t0:t0 + t] = 0
        return inputs * mask
