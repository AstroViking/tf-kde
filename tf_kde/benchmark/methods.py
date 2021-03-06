import tensorflow as tf
import numpy as np
import zfit as zfit
from KDEpy import NaiveKDE, FFTKDE
import seaborn as sns
from zfit.pdf import GaussianKDE1DimV1
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic, KernelDensityEstimationZfit, KernelDensityEstimationZfitFFT, KernelDensityEstimationZfitISJ, KernelDensityEstimationZfitHofmeyr 
from tf_kde.helper import bandwidth as bw_helper


class KDEpy: 
    description = 'KDE using KDEpy.NaiveKDE'

    def __init__(self, data, bandwidth, xlim = None):
        self._instance = NaiveKDE(kernel="gaussian", bw=bandwidth).fit(data)

    def pdf(self, x):
        x = x.numpy()
        return self._instance.evaluate(x)


class KDEpyFFT:
    description = 'KDE using KDEpy.FFTKDE'

    def __init__(self, data, bandwidth, xlim = None):
        self._instance = FFTKDE(kernel="gaussian", bw=bandwidth).fit(data)

    def pdf(self, x):
        x = x.numpy()
        return self._instance.evaluate(x)


class KDEpyFFTwithISJBandwidth:
    description = 'KDE using KDEpy.FFTKDE and ISJ bandwidth'

    def __init__(self, data, bandwidth, xlim = None):
        self._instance = FFTKDE(kernel="gaussian", bw='ISJ').fit(data)

    def pdf(self, x):
        x = x.numpy()
        return self._instance.evaluate(x)


class ZfitExact:
    description = 'KDE implemented as Zfit wrapped MixtureSameFamily'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth)

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)


class ZfitBinned:
    description = 'KDE implemented as Zfit wrapped MixtureSameFamily with linearly binned Data'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth, use_grid=True)

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)


class ZfitSimpleBinned:
    description = 'KDE implemented as Zfit wrapped MixtureSameFamily with simple binned Data'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth, use_grid=True, binning_method='simple')

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)


class ZfitFFT:
    description = 'KDE implemented as Zfit BasePdf using linear binning and the FFT algorithm'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfitFFT(obs=obs, data=data, bandwidth=bandwidth)

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)


class ZfitFFTwithISJBandwidth:
    description = 'KDE implemented as Zfit BasePdf using linear binning, the FFT algorithm and bandwith computed with the ISJ algorithm'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        bandwidth = bw_helper.improved_sheather_jones(data)
        self._instance = KernelDensityEstimationZfitFFT(obs=obs, data=data, bandwidth=bandwidth)

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)


class ZfitISJ:
    description = 'KDE implemented as Zfit BasePdf using linear binning and the ISJ algorithm'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfitISJ(obs=obs, data=data)

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)


class ZfitExactwithAdaptiveBandwidth:
    description = 'KDE implemented as Zfit wrapped MixtureSameFamily with adaptive bandwidth'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = GaussianKDE1DimV1(obs=obs, bandwidth='adaptive', data=data.astype(np.float64))

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)

class ZfitHofmeyrK1:
    description = 'KDE implemented using Kernels of the form poly(x) * exp(x)'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfitHofmeyr(obs=obs, data=data, bandwidth=bandwidth, implementation='tensorflow')

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)

class ZfitHofmeyrK1withNumpy:
    description = 'KDE implemented using Kernels of the form poly(x) * exp(x) using tf.numpy_function'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfitHofmeyr(obs=obs, data=data, bandwidth=bandwidth, implementation='numpy')

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)

class ZfitHofmeyrK1withCpp:
    description = 'KDE implemented using Kernels of the form poly(x) * exp(x) using tf.numpy_function'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        self._instance = KernelDensityEstimationZfitHofmeyr(obs=obs, data=data, bandwidth=bandwidth, implementation='cpp')

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)

class ZfitHofmeyrK1withCppAndISJBandwidth:
    description = 'KDE implemented using Kernels of the form poly(x) * exp(x) using tf.numpy_function'

    def __init__(self, data, bandwidth, xlim):
        obs = zfit.Space('x', limits=(xlim[0], xlim[1]))
        bandwidth = bw_helper.improved_sheather_jones(data)
        self._instance = KernelDensityEstimationZfitHofmeyr(obs=obs, data=data, bandwidth=bandwidth, implementation='cpp')

    @tf.function(autograph=False, experimental_relax_shapes=True)
    def pdf(self, x):
        return self._instance.pdf(x)