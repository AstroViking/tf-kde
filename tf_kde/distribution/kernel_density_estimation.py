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


class KernelDensityEstimation(MixtureSameFamily):
    """ Kernel density estimation based on data.

        Implements Linear Binning and Fast Fourier Transform to speed up the computation.
    """
    
    def __init__(self,
                 data,
                 bandwidth=0.01,
                 kernel=Normal,
                 use_grid=False,
                 use_fft=False,
                 num_grid_points=1024,
                 reparameterize=False,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='KernelDensityEstimation'):

        components_distribution_generator = lambda loc, scale: Independent(kernel(loc=loc, scale=scale))

        with tf.name_scope(name) as name:
            self._use_fft=use_fft

            dtype = dtype_util.common_dtype([bandwidth, data], tf.float32)
            self._bandwidth = tensor_util.convert_nonref_to_tensor(
                bandwidth, name='bandwidth', dtype=dtype)
            self._data = tensor_util.convert_nonref_to_tensor(
                data, name='data', dtype=dtype)
            
            if(use_fft):
                self._grid = self._generate_grid(num_grid_points)
                self._grid_data = self._linear_binning()

                mixture_distribution=Categorical(probs=self._grid_data)
                components_distribution=components_distribution_generator(loc=self._grid, scale=self._bandwidth)

            elif(use_grid):
                self._grid = self._generate_grid(num_grid_points)
                self._grid_data = self._linear_binning()

                mixture_distribution=Categorical(probs=self._grid_data)
                components_distribution=components_distribution_generator(loc=self._grid, scale=self._bandwidth)

            else:
                self._grid = None
                self._grid_data = None
                n = self._data.shape[0]
                mixture_distribution=Categorical(probs=[1 / n] * n)
                components_distribution=components_distribution_generator(loc=self._data, scale=self._bandwidth)

            super(KernelDensityEstimation, self).__init__(
                mixture_distribution=mixture_distribution, 
                components_distribution=components_distribution,
                reparameterize=reparameterize,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name
            )
    
    def _generate_grid(self, 
                       num_points): 

        minimum = tf.math.reduce_min(self._data)
        maximum = tf.math.reduce_max(self._data)
        return tf.linspace(minimum, maximum, num=num_points)
    
    def _linear_binning(self, 
                        weights=None):

        if weights is None:
            weights = tf.ones_like(self._data)

        grid_min = tf.math.reduce_min(self._grid)
        grid_max = tf.math.reduce_max(self._grid)
        num_intervals = tf.math.subtract(tf.size(self._grid), tf.constant(1))
        dx = tf.math.divide(tf.math.subtract(grid_max, grid_min), tf.cast(num_intervals, tf.float32))

        transformed_data = tf.math.divide(tf.math.subtract(self._data, grid_min), dx)

        # Compute the integral and fractional part of the data
        # The integral part is used for lookups, the fractional part is used
        # to weight the data
        integral = tf.math.floor(transformed_data)
        fractional = tf.math.subtract(transformed_data, integral)
        
        # Compute the weights for left and right side of the linear binning routine
        frac_weights = tf.math.multiply(fractional, weights)
        neg_frac_weights = tf.math.subtract(weights, frac_weights)

        # If the data is not a subset of the grid, the integral values will be
        # outside of the grid. To solve the problem, we filter these values away
        #unique_integrals = tf.unique(integral)
        #unique_integrals = unique_integrals[(unique_integrals >= 0) & (unique_integrals <= len(grid_points))]

        bincount_left = tf.roll(tf.concat(tf.math.bincount(tf.cast(integral, tf.int32), weights=frac_weights), tf.constant(0)), shift=1, axis=0)
        bincount_right = tf.math.bincount(tf.cast(integral, tf.int32), weights=neg_frac_weights)

        bincount = tf.add(bincount_left, bincount_right)

        return bincount

    def _generate_fft_grid(self,
                            num_points):

        grid_min = tf.math.reduce_min(self._data)
        grid_max = tf.math.reduce_max(self._data)
        num_intervals = tf.math.subtract(num_points, tf.constant(1))
        dx = tf.math.divide(tf.math.subtract(grid_max, grid_min), tf.cast(num_intervals, tf.float32))

        return tf.linspace(tf.math.multiply(tf.math.negative(dx), num_points), tf.math.multiply(dx, num_points), tf.math.add(tf.math.multiply(tf.cast(num_points, tf.int32), tf.constant(2)), tf.constant(1)))

    def _evaluate_fft(self, x):
        # Reshape in preparation to
        
        #! Not implemented yet!
        return super(KernelDensityEstimation, self)._log_prob(x)
        

        """num_points = tf.size(self._grid)
        x = tensor_util.convert_nonref_to_tensor(x, name='x', dtype=tf.float32)
        l = tf.linspace(tf.math.multiply(tf.math.negative(dx), num_points), tf.math.multiply(dx, num_points), tf.math.add(tf.math.multiply(tf.cast(num_points, tf.int32), tf.constant(2)), tf.constant(1)))

        components_distribution_generator = lambda loc, scale: Independent(kernel(loc=loc, scale=scale))
                
        n = x.shape[0]
        mixture_distribution=Categorical(probs=[1/n] * n)
        components_distribution=components_distribution_generator(loc=x, scale=self._bandwidth)

        super(KernelDensityEstimation, self).__init__(
                mixture_distribution=mixture_distribution, 
                components_distribution=components_distribution)
        
        super(KernelDensityEstimation, self)._log_prob(x)

        kernel_weights = super(KernelDensityEstimation, self)._log_prob(x)
        zeros_count = tf.cast(tf.math.divide(tf.math.subtract(kernel_weights.shape[0], x.shape[0]), tf.constant(2)), tf.int32)
        paddings = paddings = [[zeros_count, zeros_count]]
        data = tf.pad(x, paddings, "CONSTANT")

        tf.print(kernel_weights.shape)
        tf.print(data.shape)

        tf.print(data)

        # Use FFT
        kernel_weights = tf.signal.rfft(kernel_weights)
        data = tf.signal.rfft(data)

        return tf.signal.irfft(tf.math.multiply(data, kernel_weights))[zeros_count:-zeros_count]"""

    def _log_prob(self, x):

        if(self._use_fft):
            return self._evaluate_fft(x)
        else:
            return super(KernelDensityEstimation, self)._log_prob(x)
