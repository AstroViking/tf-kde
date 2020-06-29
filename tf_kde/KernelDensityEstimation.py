import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util

class KernelDensityEstimation(distribution.Distribution):
    """ Kernel density estimation based on data."""
    
    def __init__(self,
                 data,
                 bandwidth=0.1,
                 kernel_distribution=Normal,
                 use_FFT=True,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='KernelDensityEstimation'):

        parameters = dict(locals())
        
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([bandwidth, data], tf.float32)
            self._bandwidth = tensor_util.convert_nonref_to_tensor(
              bandwidth, name='bandwidth', dtype=dtype)
            self._data = tensor_util.convert_nonref_to_tensor(
              data, name='data', dtype=dtype)
              
            self._kernel=kernel_distribution(loc=self._data, scale=tf.broadcast_to(bandwidth, data.size))
            super(KernelDensityEstimation, self).__init__(
              dtype=dtype,
              reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return {
            'bandwidth': tf.convert_to_tensor(sample_shape, dtype=tf.float32),
            'data': [tf.convert_to_tensor(sample_shape, dtype=tf.float32)]
        }

    @classmethod
    def _params_event_ndims(cls):
        return dict(bandwidth=0, data=1)

    @property
    def bandwidth(self):
        """Input argument `bandwidth`."""
        return self._bandwidth
    
    @property
    def data(self):
        """Input argument `data`."""
        return self._data
        
    def _batch_shape_tensor(self):
        return self._categorical.batch_shape_tensor()

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self.bandwidth.shape,
            self.data.shape)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])
        
    def _batch_shape_tensor(self, bandwidth=None, data=None):
        return tf.broadcast_dynamic_shape(
            tf.shape(self.bandwidth if bandwidth is None else bandwidth),
            tf.shape(self.high if data is None else data))
            
    def _log_prob(self, x):
        return tf.reduce_logsumexp(self._kernel._log_prob(x)
                               +self._w_lp-self._w_norm_lp, [-1],
                               keep_dims=True)

    def _prob(self, x):
        return tf.exp(self._log_prob(x))


    def _log_cdf(self, x):
        return tf.reduce_logsumexp(self._kernel._log_cdf(x)
                               +self._w_lp-self._w_norm_lp, [-1],
                               keep_dims=True)

    def _cdf(self, x):
        return tf.exp(self._log_cdf(x))
    
    """
      
    def _sample_n(self, n, seed=None):
        bandwidth = tf.convert_to_tensor(self.bandwidth)
        data = tf.convert_to_tensor(self.data)
        shape = tf.concat([[n], self._batch_shape_tensor(
        bandwidth=bandwidth, data=data)], 0)
        
        MixtureSameFamily()
        
        fac = tf.constant(1.0 / np.sqrt(2.0 * np.pi), tf.float32)
        exp_fac = tf.constant(-1.0/2.0, tf.float32)
        y_fac = tf.constant(1.0/(h1 * n_datapoints), tf.float32)
        h = tf.constant(bandwidth, tf.float32)
        
        tf.math.divide(tf.math.subtract(x, data), bandwidth)
        
  
        gauss_kernel = lambda x: tf.math.multiply(fac, tf.math.exp(tf.math.multiply(exp_fac, tf.math.square(x))))
        calc_value = lambda x: tf.math.multiply(y_fac, tf.math.reduce_sum(gauss_kernel(tf.math.divide(tf.math.subtract(x, data), h))))
      
    return low + self._range(low=low, high=high) * samples

  def _prob(self, x):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    return tf.where(
        tf.math.is_nan(x),
        x,
        tf.where(
            # This > is only sound for continuous uniform
            (x < low) | (x > high),
            tf.zeros_like(x),
            tf.ones_like(x) / self._range(low=low, high=high)))

  def _cdf(self, x):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(x), self._batch_shape_tensor(low=low, high=high))
    zeros = tf.zeros(broadcast_shape, dtype=self.dtype)
    ones = tf.ones(broadcast_shape, dtype=self.dtype)
    result_if_not_big = tf.where(x < low, zeros,
                                 (x - low) / self._range(low=low, high=high))
    return tf.where(x >= high, ones, result_if_not_big)

  def _quantile(self, value):
    return (1. - value) * self.low + value * self.high

  def _entropy(self):
    return tf.math.log(self._range())

  def _mean(self):
    return (self.low + self.high) / 2.

  def _variance(self):
    return tf.square(self._range()) / 12.

  def _stddev(self):
    return self._range() / np.sqrt(12.)

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(
        low=self.low, high=self.high, validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    low = None
    high = None
    if is_init != tensor_util.is_ref(self.low):
      low = tf.convert_to_tensor(self.low)
      high = tf.convert_to_tensor(self.high)
      assertions.append(assert_util.assert_less(
          low, high, message='uniform not defined when `low` >= `high`.'))
    if is_init != tensor_util.is_ref(self.high):
      low = tf.convert_to_tensor(self.low) if low is None else low
      high = tf.convert_to_tensor(self.high) if high is None else high
      assertions.append(assert_util.assert_less(
          low, high, message='uniform not defined when `low` >= `high`.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_greater_equal(
        x, self.low, message='Sample must be greater than or equal to `low`.'))
    assertions.append(assert_util.assert_less_equal(
        x, self.high, message='Sample must be less than or equal to `high`.'))
    return assertions
"""
