import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
from tf_kde.distribution import KernelDensityEstimation

r_seed = 1978239485
n_datapoints = 1000000
tfd = tfp.distributions

mix_3gauss_1exp_1uni = tfd.Mixture(

  cat=tfd.Categorical(probs=[0.1, 0.2, 0.1, 0.4, 0.2]),

  components=[
    tfd.Normal(loc=-1., scale=0.4),
    tfd.Normal(loc=+1., scale=0.5),
    tfd.Normal(loc=+1., scale=0.3),
    tfd.Exponential(rate=2),
    tfd.Uniform(low=-5, high=5)
])

data = mix_3gauss_1exp_1uni.sample(sample_shape=n_datapoints, seed=r_seed)
data = data.numpy()

#n_testpoints = 200

#dist = KernelDensityEstimation(bandwidth=0.01, data=data)

#x = np.linspace(-5.0, 5.0, num=n_testpoints, dtype=np.float32)
#y = dist.prob(x)
#print(y)

#dist2 = KernelDensityEstimation(bandwidth=0.01, data=data, kernel='epanechnikov')
#y2 = dist2.prob(x)
#print(y2)


#f = lambda x: tfd.Independent(tfd.Normal(loc=data, scale=1.))
#n = tf.cast(tf.size(data), tf.float32)
#kde = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=[1 / n] * n), components_distribution=f(data))

#y3 = kde._prob(x)

#print(y3)