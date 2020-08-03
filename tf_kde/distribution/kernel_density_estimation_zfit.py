from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd

from zfit.pdf import WrapDistribution, BasePDF
from zfit import z, ztypes
from zfit.core.interfaces import ZfitData, ZfitSpace
from zfit.util import ztyping
from zfit.util.exception import OverdefinedError, ShapeIncompatibleError
from zfit.core.space import supports, Space

from tf_kde.helper import binning as binning_helper
from tf_kde.helper import convolution as convolution_helper
from tf_kde.helper import bandwidth as bandwidth_helper
from tf_kde.helper import improved_sheather_jones as isj_helper


class KernelDensityEstimation(WrapDistribution):
    _N_OBS = 1

    def __init__(self, 
                 obs: ztyping.ObsTypeInput, 
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel = tfd.Normal,
                 support = None,
                 use_grid = False,
                 num_grid_points = 1024,
                 binning_method = 'linear',
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

        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._bandwidth = tf.convert_to_tensor(bandwidth, ztypes.float)
        self._kernel = kernel
        self._weights = weights
        self._grid = None
        self._grid_data = None

        if use_grid:
            self._grid = binning_helper.generate_grid(self._data, num_grid_points=self._num_grid_points)
            self._grid_data = binning_helper.bin(self._binning_method, self._data, self._grid, self._weights)

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


class KernelDensityEstimationFFT(BasePDF):
    _N_OBS = 1

    def __init__(self, 
                 obs: ztyping.ObsTypeInput, 
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel = tfd.Normal,
                 support = None,
                 num_grid_points = 1024,
                 binning_method = 'linear',
                 fft_method = 'conv1d',
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KernelDensityEstimationFFT"):
        r"""
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.
        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        It is computed by using a convolution of the data with the kernels evaluated at fixed grid points and then 
        interpolating between this points to get an estimate for x.

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

        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._fft_method = fft_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._bandwidth = tf.convert_to_tensor(bandwidth, ztypes.float)
        self._kernel = kernel
        self._weights = weights
        self._support = support
        self._grid = None
        self._grid_data = None

        self._grid = binning_helper.generate_grid(self._data, num_grid_points=self._num_grid_points)
        self._grid_data = binning_helper.bin(self._binning_method, self._data, self._grid, self._weights)
        self._grid_convolved_data = convolution_helper.convolve_data_with_kernel(self._kernel, self._bandwidth, self._grid_data, self._grid, self._support, self._fft_method)

        params = {'bandwidth': self._bandwidth}
        super().__init__(obs=obs, name=name, params=params)

    def _unnormalized_pdf(self, x, norm_range=False):

        x = z.unstack_x(x)
        x_min = tf.reduce_min(self._grid)
        x_max = tf.reduce_max(self._grid)

        return tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_convolved_data)


class KernelDensityEstimationISJ(BasePDF):
    _N_OBS = 1

    def __init__(self, 
                 obs: ztyping.ObsTypeInput, 
                 data: ztyping.ParamTypeInput,
                 num_grid_points = 1024,
                 binning_method = 'linear',
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KernelDensityEstimationISJ"):
        r"""
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.
        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        It is computed by using a trick described in a paper by Botev et al. that uses the fact, that the Kernel Density Estimation
        with a Gaussian Kernel is a solution to the Heat Euqation.

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

        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._weights = weights
        self._grid = None
        self._grid_data = None

        self._bandwidth, self._grid_data, self._grid = isj_helper.calculate_bandwidth_and_density(self._data, self._num_grid_points, self._binning_method, self._weights)

        params = {}
        super().__init__(obs=obs, name=name, params=params)

    def _unnormalized_pdf(self, x, norm_range=False):

        x = z.unstack_x(x)
        x_min = tf.reduce_min(self._grid)
        x_max = tf.reduce_max(self._grid)

        return tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_data)

