import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd

gaussian = tfd.Normal(loc=0., scale=1.)

uniform = tfd.Uniform(low=-2., high=2.)

bimodal = tfd.Mixture(

  cat=tfd.Categorical(probs=[1/2, 1/2]),

  components=[
    tfd.Normal(loc=-5, scale=1.),
    tfd.Normal(loc=5, scale=1.)
])

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

asymmetric_double_claw = tfd.Mixture(

  cat=tfd.Categorical(probs=[46./100., 46./100, 1./300., 1./300., 1./300., 7./300., 7./300., 7./300.]),

  components=[
    tfd.Normal(loc=-1., scale=2./3.),
    tfd.Normal(loc=1., scale=2./3.),

    tfd.Normal(loc=-1./2., scale=0.01),
    tfd.Normal(loc=-1., scale=0.01),
    tfd.Normal(loc=-3./2., scale=0.01),

    tfd.Normal(loc=1./2., scale=0.07),
    tfd.Normal(loc=1., scale=0.07),
    tfd.Normal(loc=3./2., scale=0.07)
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