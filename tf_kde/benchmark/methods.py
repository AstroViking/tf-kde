import tensorflow as tf
import numpy as np
import zfit as zfit
from KDEpy import NaiveKDE, FFTKDE
import seaborn as sns
from zfit.pdf import GaussianKDE1DimV1
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic, KernelDensityEstimationZfit, KernelDensityEstimationZfitFFT, KernelDensityEstimationZfitISJ 
from tf_kde.helper import bandwidth as bw_helper


labels = {
    'kdepy': 'KDE using KDEpy.NaiveKDE',
    'kdepy_fft': 'KDE using KDEpy.FFTKDE',
    'kdepy_fft_isj': 'KDE using KDEpy.FFTKDE and ISJ bandwidth',
    'zfit_mixture': 'KDE implemented as Zfit wrapped TensorFlow Probability class',
    'zfit_binned': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Linear Binned Data',
    'zfit_simple_binned': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Simple Binned Data',
    'zfit_fft': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT',
    'zfit_ffts': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT with tf.signal.fft',
    'zfit_fft_with_isj_bandwidth': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT and Bandwith computed with ISJ',
    'zfit_isj': 'KDE implemented as Zfit wrapped ISJ method',
    'zfit_adaptive': 'KDE implemented as Zfit wrapped MixtureSameFamily with adaptive Bandwith calculation',
}

def kdepy(data, x, bandwidth):
    x = np.array(x)
    return NaiveKDE(kernel="gaussian", bw=bandwidth).fit(data).evaluate(x)

def kdepy_fft(data, x, bandwidth):
    x = np.array(x)
    return FFTKDE(kernel="gaussian", bw=bandwidth).fit(data).evaluate(x)

def kdepy_fft_isj(data, x, bandwidth):
    x = np.array(x)
    return FFTKDE(kernel="gaussian", bw='ISJ').fit(data).evaluate(x)

@tf.function(autograph=False)
def zfit_mixture_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth)
    return dist.pdf(x)

def zfit_mixture(data, x, bandwidth):
    return zfit_mixture_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_binned_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth, use_grid=True)
    return dist.pdf(x)  

def zfit_binned(data, x, bandwidth):
    return zfit_binned_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_simple_binned_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth, use_grid=True, binning_method='simple')
    return dist.pdf(x)  

def zfit_simple_binned(data, x, bandwidth):
    return zfit_simple_binned_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_fft_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfitFFT(obs=obs, data=data, bandwidth=bandwidth)
    return dist.pdf(x)

def zfit_fft(data, x, bandwidth):
    return zfit_fft_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_ffts_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfitFFT(obs=obs, data=data, bandwidth=bandwidth, fft_method='signal')
    return dist.pdf(x)

def zfit_ffts(data, x, bandwidth):
    return zfit_ffts_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_fft_with_isj_bandwidth_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))

    bandwidth = bw_helper.improved_sheather_jones(data)
    dist = KernelDensityEstimationZfitFFT(obs=obs, data=data, bandwidth=bandwidth)
    return dist.pdf(x)

def zfit_fft_with_isj_bandwidth(data, x, bandwidth):
    return zfit_fft_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_isj_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfitISJ(obs=obs, data=data)
    return dist.pdf(x)

def zfit_isj(data, x, bandwidth):
    return zfit_isj_internal(data.astype(np.float64), x, bandwidth).numpy()

def zfit_adaptive_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = GaussianKDE1DimV1(obs=obs, bandwidth='adaptive', data=data)
    return dist.pdf(x)

def zfit_adaptive(data, x, bandwidth):
    return zfit_adaptive_internal(data.astype(np.float64), x, bandwidth).numpy()