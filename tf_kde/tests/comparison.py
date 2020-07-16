import tensorflow as tf
import numpy as np
import zfit as zfit
from KDEpy import NaiveKDE, FFTKDE
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic, KernelDensityEstimationZfit
from zfit_benchmark.timer import Timer
import seaborn as sns
import pandas as pd

from tf_kde.tests.test_distribution import data

n_testpoints = 1024
run_twice = True
methods_to_evaluate = [
    'basic',
    'FFTKDEpy',
    'zfitFFT',
    'zfitFFTs'
]

def kde_basic(data, x):

    fac = 1.0 / np.sqrt(2.0 * np.pi)
    exp_fac = -1.0/2.0
    h = 0.01
    y_fac = 1.0/(h*data.size)

    gauss_kernel = lambda x: fac * np.exp(exp_fac * x**2)
          
    y = np.zeros(x.size)

    for i, x_i in enumerate(x):
        y[i] = y_fac * np.sum(gauss_kernel((x_i-data)/h))
      
    return y
  
def kde_seaborn(data, x):
    sns.distplot(data, bins=1000, kde=True, rug=False)
    return np.NaN


def kde_kdepy(data, x):
    x = np.array(x)
    return NaiveKDE(kernel="gaussian").fit(data).evaluate(x)

def kde_kdepy_fft(data, x):
    x = np.array(x)
    return FFTKDE(kernel="gaussian", bw="silverman").fit(data).evaluate(x)

@tf.function(autograph=False)
def kde_basic_tf_internal(data, x, n_datapoints):

    h1 = 0.01
  
    fac = tf.constant(1.0 / np.sqrt(2.0 * np.pi), tf.float32)
    exp_fac = tf.constant(-1.0/2.0, tf.float32)
    y_fac = tf.constant(1.0/(h1 * n_datapoints), tf.float32)
    h = tf.constant(h1, tf.float32)
  
    gauss_kernel = lambda x: tf.math.multiply(fac, tf.math.exp(tf.math.multiply(exp_fac, tf.math.square(x))))
    calc_value = lambda x: tf.math.multiply(y_fac, tf.math.reduce_sum(gauss_kernel(tf.math.divide(tf.math.subtract(x, data), h))))
  
    return tf.map_fn(calc_value, x)

def kde_basic_tf(data, x):
    n_datapoints = data.size
    return kde_basic_tf_internal(data, x, n_datapoints).numpy()

@tf.function(autograph=False)
def kde_tfp_internal(data, x):
    dist = KernelDensityEstimationBasic(bandwidth=0.01, data=data)
    return dist.prob(x)

def kde_tfp(data, x):
    return kde_tfp_internal(data, x).numpy()

@tf.function(autograph=False)
def kde_tfp_mixture_internal(data, x):
    dist = KernelDensityEstimation(bandwidth=0.01, data=data)
    return dist.prob(x)

def kde_tfp_mixture(data, x):
    return kde_tfp_mixture_internal(data, x).numpy()

@tf.function(autograph=False)
def kde_tfp_mixture_binned_internal(data, x):
    dist = KernelDensityEstimation(bandwidth=0.01, data=data, use_grid=True)
    return dist.prob(x)

def kde_tfp_mixture_binned(data, x):
    return kde_tfp_mixture_binned_internal(data, x).numpy()

@tf.function(autograph=False)
def kde_zfit_internal(data, x):
    obs = zfit.Space('x', limits=(np.min(x), np.max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=0.01)
    return dist.pdf(x)

def kde_zfit(data, x):
    return kde_zfit_internal(data.astype(np.float64), x).numpy()

@tf.function(autograph=False)
def kde_zfit_binned_internal(data, x):
    obs = zfit.Space('x', limits=(np.min(x), np.max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, bandwidth=0.01, use_grid=True)
    return dist.pdf(x)  

def kde_zfit_binned(data, x):
    return kde_zfit_binned_internal(data.astype(np.float64), x).numpy()

@tf.function(autograph=False)
def kde_zfit_fft_internal(data, x):
    obs = zfit.Space('x', limits=(np.min(x), np.max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, num_grid_points=1024, bandwidth=0.01, use_fft=True)
    return dist.pdf(x)

def kde_zfit_fft(data, x):
    return kde_zfit_fft_internal(data.astype(np.float64), x).numpy()

@tf.function(autograph=False)
def kde_zfit_ffts_internal(data, x):
    obs = zfit.Space('x', limits=(np.min(x), np.max(x)))
    dist = KernelDensityEstimationZfit(obs=obs, data=data, num_grid_points=1024, bandwidth=0.01, use_fft=True, fft_method='signal')
    return dist.pdf(x)

def kde_zfit_ffts(data, x):
    return kde_zfit_ffts_internal(data.astype(np.float64), x).numpy()

methods = pd.DataFrame({
    'identifier': [
        'basic',
        'seaborn',
        'FFTKDEpy',
        'basicTF',
        'tfp',
        'tfpM',
        'tfpMB',
        'zfit',
        'zfitB',
        'zfitFFT',
        'zfitFFTs'
    ],
    'label': [
        'Basic KDE with Python',
        'KDE using seaborn.distplot',
        'KDE using KDEpy.FFTKDE',
        'Basic KDE in TensorFlow',
        'KDE implemented as TensorFlow Probability Distribution Subclass',
        'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass',
        'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass with Binned Data',
        'KDE implemented as Zfit wrapped TensorFlow Probability class',
        'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data',
        'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT',
        'KDE implemented as Zfit wrapped TensorFlow Probability class with Binned Data and FFT with tf.signal.fft'
    ],
    'function':[
        kde_basic,
        kde_seaborn,
        kde_kdepy_fft,
        kde_basic_tf,
        kde_tfp,
        kde_tfp_mixture,
        kde_tfp_mixture_binned,
        kde_zfit,
        kde_zfit_binned,
        kde_zfit_fft,
        kde_zfit_ffts
    ]
})
methods.set_index('identifier', drop=False, inplace=True)
methods['runtime'] = np.NaN

estimations = pd.DataFrame()
estimations['x'] = np.linspace(-7.0, 7.0, num=n_testpoints, dtype=np.float32)

for index, method in methods.iterrows():
    if method['identifier'] in methods_to_evaluate:
        with Timer('Benchmarking') as timer:
            estimations[method['identifier']] = method['function'](data, estimations['x'])
            timer.stop()
        methods.at[method['identifier'], 'runtime'] = timer.elapsed
        if run_twice:
            with Timer('Benchmarking') as timer:
                estimations[method['identifier']] = method['function'](data, estimations['x'])
                timer.stop()
            methods.at[method['identifier'], 'runtime_second'] = timer.elapsed
    
    

methods.drop('function', axis=1, inplace=True)


print(methods)
print(estimations)

import matplotlib.pyplot as plt
estimations.plot(x = 'x', y = methods_to_evaluate)
plt.show()
