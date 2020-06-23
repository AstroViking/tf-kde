import tensorflow as tf
import tensorflow_probability as tfp

class KernelDensityEstimation(tfp.distributions.Distribution):
  """ Kernel density estimation based on data. """ 
