import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import Categorical
from tensorflow_probability.python.distributions import Independent
from tensorflow_probability.python.distributions import MixtureSameFamily
from tensorflow_probability.python.distributions import Normal
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


class KernelDensityEstimationBasic(distribution.Distribution):
    """ Kernel density estimation based on data."""
    
    def __init__(self,
                 data,
                 bandwidth=0.1,
                 kernel='gaussian',
                 use_FFT=False,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='KernelDensityEstimationBasic'):

        self._kernels = {
            'gaussian': lambda x: tf.math.multiply(tf.constant(1.0 / np.sqrt(2.0 * np.pi), tf.float32), tf.math.exp(tf.math.multiply(tf.constant(-1.0/2.0, tf.float32), tf.math.square(x)))),
            #!TODO Fix epanechnikov kernel!
            'epanechnikov': lambda x: tf.math.multiply(tf.constant(3.0 / 4.0, tf.float32), tf.math.subtract(tf.constant(1.0, tf.float32), tf.math.square(x)))
        }

        self._kernel_name=kernel

        parameters = dict(locals())
        
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([bandwidth, data], tf.float32)
            self._bandwidth = tensor_util.convert_nonref_to_tensor(
              bandwidth, name='bandwidth', dtype=dtype)
            self._data = tensor_util.convert_nonref_to_tensor(
              data, name='data', dtype=dtype)
             
            self._use_FFT=use_FFT
              
            super(KernelDensityEstimationBasic, self).__init__(
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
      return tf.math.log(self._prob(x)) 

    def _prob(self, x):

        if self._use_FFT == False:
            calc_value = lambda x: tf.math.multiply(tf.constant(1.0)/(self._bandwidth * tf.cast(tf.size(self._data), tf.float32)), tf.math.reduce_sum(self._kernels[self._kernel_name](tf.math.divide(tf.math.subtract(x, self._data), self._bandwidth))))  
            results = tf.map_fn(calc_value, x)
        else:
          results = tf.zeros(tf.size(self._data))
            
        return results