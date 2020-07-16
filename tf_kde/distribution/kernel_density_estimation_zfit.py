#  Copyright (c) 2020 zfit
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd

from zfit.models.dist_tfp import WrapDistribution
from zfit import z, ztypes
from zfit.core.interfaces import ZfitData, ZfitSpace
from zfit.util import ztyping
from zfit.util.exception import OverdefinedError, ShapeIncompatibleError
from zfit.core.space import supports, Space

class KernelDensityEstimation(WrapDistribution):
    _N_OBS = 1

    def __init__(self, 
                 obs: ztyping.ObsTypeInput, 
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel = tfd.Normal,
                 support = None,
                 use_grid = False,
                 use_fft = False,
                 num_grid_points = 1024,
                 binning_method = 'linear',
                 fft_method = 'conv1d',
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KernelDensityEstimation"):
        r"""
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.
        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        Args:
            data: 1-D Tensor-like.
            bandwidth: Bandwidth of the kernel. Valid options are {'silverman', 'scott', 'adaptiveV1'} or a numerical.
                If a numerical is given, it as to be broadcastable to the batch and event shape of the distribution.
                A scalar or a `zfit.Parameter` will simply broadcast to `data` for a 1-D distribution.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """

        if isinstance(data, ZfitData):
            if data.weights is not None:
                if weights is not None:
                    raise OverdefinedError("Cannot specify weights and use a `ZfitData` with weights.")
                else:
                    weights = data.weights

            if data.n_obs > 1:
                raise ShapeIncompatibleError(f"KDE is 1 dimensional, but data {data} has {data.n_obs} observables.")
            data = z.unstack_x(data)

        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], ztypes.float)

        components_distribution_generator = lambda loc, scale: tfd.Independent(kernel(loc=loc, scale=scale))

        self._use_fft = use_fft
        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._fft_method = fft_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._bandwidth = tf.convert_to_tensor(bandwidth, ztypes.float)
        self._kernel = kernel
        self._support = support
        self._weights = weights
        self._grid = None
        self._grid_data = None

        # If FFT is used, we delay the parent initialization until later
        if self._use_fft:

            self._grid = self._generate_grid(self._data, num_grid_points=self._num_grid_points)
            self._grid_data = self._linear_binning(self._data, self._grid, self._weights)
            self._grid_fft_data = self._generate_convolved_data(self._grid_data, self._grid)

            params = {'bandwidth': self._bandwidth}
            super(WrapDistribution, self).__init__(obs=obs, name=name, params=params)
        else:
            if use_grid:
                self._grid = self._generate_grid(self._data, num_grid_points=self._num_grid_points)
                self._grid_data = self._linear_binning(self._data, self._grid, self._weights)

                mixture_distribution = tfd.Categorical(probs=self._grid_data)
                components_distribution = components_distribution_generator(loc=self._grid, scale=self._bandwidth)
            
            else:
                
                if weights is not None:
                    probs = weights / tf.reduce_sum(weights)
                else:
                    probs = tf.broadcast_to(1 / size, shape=(tf.cast(size, ztypes.int),))
                
                mixture_distribution = tfd.Categorical(probs=probs)
                components_distribution = components_distribution_generator(loc=self._data, scale=self._bandwidth)

            dist_kwargs = lambda: dict(mixture_distribution=mixture_distribution,
                                    components_distribution=components_distribution)
            distribution = tfd.MixtureSameFamily

            params = {'bandwidth': self._bandwidth}

            super().__init__(obs=obs,
                            params=params,
                            dist_params={},
                            dist_kwargs=dist_kwargs,
                            distribution=distribution,
                            name=name)

    def _generate_grid(self, data, num_grid_points): 
        minimum = tf.math.reduce_min(data)
        maximum = tf.math.reduce_max(data)
        return tf.linspace(minimum, maximum, num=num_grid_points)

    def _binning(self, data, grid, weights):

        if self.binning_method == 'simple':
            return self._simple_binning(data, grid, weights)
        else:
            return self._linear_binning(data, grid, weights)

    def _simple_binning(self, data, grid, weights):

        minimum = tf.math.reduce_min(grid)
        maximum = tf.math.reduce_max(grid)
        bin_count = tf.size(grid)

        bincount = tf.histogram_fixed_width(data, [minimum, maximum], bin_count)

        #!TODO include weights in calculation

        return bincount

    
    def _linear_binning(self, data, grid, weights):

        if weights is None:
            weights = tf.ones_like(data, ztypes.float)

        weights = weights / tf.reduce_sum(weights)

        grid_min = tf.math.reduce_min(grid)
        grid_max = tf.math.reduce_max(grid)
        num_intervals = tf.math.subtract(tf.size(grid), tf.constant(1))
        dx = tf.math.divide(tf.math.subtract(grid_max, grid_min), tf.cast(num_intervals, ztypes.float))

        transformed_data = tf.math.divide(tf.math.subtract(data, grid_min), dx)

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
        #unique_integrals = np.unique(integral)
        #unique_integrals = unique_integrals[(unique_integrals >= 0) & (unique_integrals <= len(grid_points))]

        #tf.math.bincount only works with tf.int32
        bincount_left = tf.roll(tf.concat(tf.math.bincount(tf.cast(integral, tf.int32), weights=frac_weights), tf.constant(0)), shift=1, axis=0)
        bincount_right = tf.math.bincount(tf.cast(integral, tf.int32), weights=neg_frac_weights)

        bincount = tf.cast(tf.add(bincount_left, bincount_right), ztypes.float)

        return bincount

    def _generate_convolved_data(self, data, grid):

        kernel_grid_min = tf.math.reduce_min(self._grid)
        kernel_grid_max = tf.math.reduce_max(self._grid)

        num_grid_points = self._num_grid_points
        num_intervals = num_grid_points - tf.constant(1, ztypes.int)
        space_width = kernel_grid_max - kernel_grid_min
        dx = space_width / tf.cast(num_intervals, ztypes.float)
            
        L = tf.cast(num_intervals, ztypes.float)

        if self._support:
            L = tf.math.floor(tf.minimum(tf.cast(self._support, ztypes.float) * self._bandwidth * (tf.cast(num_grid_points, ztypes.float) - tf.constant(1, ztypes.float)) / (space_width),  L))

        # Calculate the kernel weights
        kernel_grid = tf.linspace(tf.constant(0, ztypes.float), dx * L, tf.cast(L, ztypes.int) + tf.constant(1, ztypes.int))
        kernel_weights = self._kernel(loc=0, scale=self._bandwidth).prob(kernel_grid)
        kernel_weights = tf.concat([tf.reverse(kernel_weights, axis = [0])[:-1], kernel_weights], axis=0)

        c = data
        k = kernel_weights

        if self._fft_method  == 'conv1d':
            c_size = tf.size(c, ztypes.int)
            c = tf.reshape(c, [1, c_size, 1], name='c')
            
            k_size = tf.size(k, ztypes.int)
            k = tf.reshape(k, [k_size, 1, 1], name='k')

            return tf.squeeze(tf.nn.conv1d(c, k, 1, 'SAME')) 
        
        else:     

            P = tf.math.pow(tf.constant(2, ztypes.int), tf.cast(tf.math.ceil(tf.math.log(tf.constant(3.0, ztypes.float) * tf.cast(num_grid_points, ztypes.float) - tf.constant(1.0, ztypes.float)) / tf.math.log(tf.constant(2.0, ztypes.float))), ztypes.int))

            right_padding = tf.cast(P, ztypes.int) - tf.constant(2, ztypes.int) * num_grid_points - tf.constant(1, ztypes.int)
            left_padding = num_grid_points - tf.constant(1, ztypes.int)

            c = tf.pad(data, [[left_padding, right_padding]])
            k = tf.pad(kernel_weights, [[0, right_padding]])

            result = tf.signal.irfft(tf.signal.rfft(c) * tf.signal.rfft(k))
            start, end = tf.constant(2, ztypes.int) * num_grid_points - tf.constant(1, ztypes.int), tf.constant(3, ztypes.int) * num_grid_points - tf.constant(1, ztypes.int)
            result = result[start:end]     

            return result


    def _unnormalized_pdf(self, x, norm_range=False):

        if self._use_fft:

            x = z.unstack_x(x)
            x_min = tf.reduce_min(self._grid)
            x_max = tf.reduce_max(self._grid)

            return tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_fft_data)

        else:
            return super()._unnormalized_pdf(x, norm_range)
            #value = z.unstack_x(x)  # TODO: use this? change shaping below?
            #return self.distribution.prob(value=value, name="unnormalized_pdf")
    
    @supports()
    def _analytic_integrate(self, limits, norm_range):
        return tf.constant(1.0, ztypes.float)

    # TODO: register integral?
    #@supports()
    #def _analytic_integrate(self, limits, norm_range):

     #   if self._use_fft:
     #       return tf.constant(1.0, ztypes.float)
     #   else:
            # Why does this not work?
            #return super()._analytic_integrate(limits, None)
    #        lower, upper = limits._rect_limits_tf
    #        lower = z.unstack_x(lower)
    #        upper = z.unstack_x(upper)
    #        tf.debugging.assert_all_finite((lower, upper), "Are infinite limits needed? Causes troubles with NaNs")
    #        return self.distribution.cdf(upper) - self.distribution.cdf(lower)

    #def _analytic_sample(self, n, limits):
    #    if self._use_fft:
    #       return super(WrapDistribution, self)._analytic_sample(n, limits)
    #    else:
    #        return super()._analytic_sample(n, limits)
        #lower, upper = limits._rect_limits_tf
        #return tfd_analytic_sample(n=n, dist=self.distribution, limits=limits)