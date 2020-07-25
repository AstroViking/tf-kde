import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd

r_seed = 1978239485
n_datapoints = 1000000
tfd = tfp.distributions

gaussian = tfd.Normal(loc=0., scale=1.)

uniform = tfd.Uniform(low=-2., high=2.)

skewed_bimodal = tfd.Mixture(

  cat=tfd.Categorical(probs=[3/4, 1/4]),

  components=[
    tfd.Normal(loc=0., scale=1.),
    tfd.Normal(loc=1.5, scale=1./3.)
])

claw = tfd.Mixture(

  cat=tfd.Categorical(probs=[1./2., 1./10, 1./10., 1./10., 1./10., 1./10.]),

  components=[
    tfd.Normal(loc=0., scale=1),
    tfd.Normal(loc=-1., scale=0.1),
    tfd.Normal(loc=-0.5, scale=0.1),
    tfd.Normal(loc=0., scale=0.1),
    tfd.Normal(loc=0.5, scale=0.1),
    tfd.Normal(loc=1., scale=0.1)
])

mix_3gauss_1exp_1uni = tfd.Mixture(

  cat=tfd.Categorical(probs=[0.1, 0.2, 0.1, 0.4, 0.2]),

  components=[
    tfd.Normal(loc=-1., scale=0.4),
    tfd.Normal(loc=+1., scale=0.5),
    tfd.Normal(loc=+1., scale=0.3),
    tfd.Exponential(rate=2),
    tfd.Uniform(low=-5, high=5)
])