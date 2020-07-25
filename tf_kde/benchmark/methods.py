import tensorflow as tf
import numpy as np
import zfit as zfit
from KDEpy import NaiveKDE, FFTKDE
import seaborn as sns
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic, KernelDensityEstimationZfit

labels = {
    'basic': 'Basic KDE with Python',
    'seaborn': 'KDE using seaborn.distplot',
    'kdepy_fft': 'KDE using KDEpy.FFTKDE',
    'tf_basic': 'Basic KDE in TensorFlow',
    'tfp': 'KDE implemented as TensorFlow Probability Distribution Subclass',
    'tfp_mixture': 'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass',
    'tfp_mixture_binned': 'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass with Binned Data',
    'zfit_mixture': 'KDE implemented as Zfit wrapped TensorFlow Probability class',
    'zfit_binned': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data',
    'zfit_fft': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT',
    'zfit_ffts': 'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT with tf.signal.fft'
}

def basic(data, x, bandwidth):

    fac = 1.0 / np.sqrt(2.0 * np.pi)
    exp_fac = -1.0/2.0
    h = bandwidth
    y_fac = 1.0/(h*data.size)

    gauss_kernel = lambda x: fac * np.exp(exp_fac * x**2)
          
    y = np.zeros(x.size)

    for i, x_i in enumerate(x):
        y[i] = y_fac * np.sum(gauss_kernel((x_i-data)/h))
      
    return y
  
def seaborn(data, x, bandwidth):
    sns.distplot(data, bins=1000, kde=True, rug=False)
    return np.NaN


def kdepy(data, x, bandwidth):
    x = np.array(x)
    return NaiveKDE(kernel="gaussian", bw=bandwidth).fit(data).evaluate(x)

def kdepy_fft(data, x, bandwidth):
    x = np.array(x)
    return FFTKDE(kernel="gaussian", bw=bandwidth).fit(data).evaluate(x)

@tf.function(autograph=False)
def tf_basic_internal(data, x, n_datapoints, bandwidth):
  
    fac = tf.constant(1.0 / np.sqrt(2.0 * np.pi), tf.float32)
    exp_fac = tf.constant(-1.0/2.0, tf.float32)
    y_fac = tf.constant(1.0/(bandwidth * n_datapoints), tf.float32)
    h = tf.constant(bandwidth, tf.float32)
  
    gauss_kernel = lambda x: tf.math.multiply(fac, tf.math.exp(tf.math.multiply(exp_fac, tf.math.square(x))))
    calc_value = lambda x: tf.math.multiply(y_fac, tf.math.reduce_sum(gauss_kernel(tf.math.divide(tf.math.subtract(x, data), h))))
  
    return tf.map_fn(calc_value, x)

def tf_basic(data, x, bandwidth):
    n_datapoints = data.size
    return tf_basic_internal(data, x, n_datapoints, bandwidth).numpy()

@tf.function(autograph=False)
def tfp_internal(data, x, bandwidth):
    dist = KernelDensityEstimationBasic(bandwidth=bandwidth, data=data)
    return dist.prob(x)

def tfp(data, x):
    return tfp_internal(data, x, bandwidth).numpy()

@tf.function(autograph=False)
def tfp_mixture_internal(data, x, bandwidth):
    dist = KernelDensityEstimation(bandwidth=bandwidth, data=data)
    return dist.prob(x)

def tfp_mixture(data, x, bandwidth):
    return tfp_mixture_internal(data, x, bandwidth).numpy()

@tf.function(autograph=False)
def tfp_mixture_binned_internal(data, x, bandwidth):
    dist = KernelDensityEstimation(bandwidth=bandwidth, data=data, use_grid=True)
    return dist.prob(x)

def tfp_mixture_binned(data, x, bandwidth):
    return tfp_mixture_binned_internal(data, x, bandwidth).numpy()

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
def zfit_fft_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth, use_fft=True)
    return dist.pdf(x)

def zfit_fft(data, x, bandwidth):
    return zfit_fft_internal(data.astype(np.float64), x, bandwidth).numpy()

@tf.function(autograph=False)
def zfit_ffts_internal(data, x, bandwidth):
    obs = zfit.Space('x', limits=(tf.reduce_min(x), tf.reduce_max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=bandwidth, use_fft=True, fft_method='signal')
    return dist.pdf(x)

def zfit_ffts(data, x, bandwidth):
    return zfit_ffts_internal(data.astype(np.float64), x, bandwidth).numpy()