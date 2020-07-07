import tensorflow as tf
import numpy as np
from KDEpy import FFTKDE
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic
from zfit_benchmark.timer import Timer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from test_distribution import data


n_testpoints = 200

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

def kde_kdepy_fft(data, x):
    x = np.array(x)
    return FFTKDE(kernel="gaussian", bw="silverman").fit(data).evaluate(x)

@tf.function(autograph=False)
def kde_basic_tf_internal(data, x, n_datapoints):

    # TODO: Use tf-kde package here
  
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

def kde_tfp(data, x):
    dist = KernelDensityEstimationBasic(bandwidth=0.01, data=data)
    return dist.prob(x).numpy()

def kde_tfp_mixture(data, x):
    dist = KernelDensityEstimation(bandwidth=0.01, data=data)
    return dist.prob(x).numpy()

def kde_tfp_mixture_with_binned_data(data, x):
    dist = KernelDensityEstimation(bandwidth=0.01, data=data, use_grid=True)
    return dist.prob(x).numpy()

def kde_tfp_mixture_with_fft(data, x):
    dist = KernelDensityEstimation(bandwidth=0.01, data=data, use_grid=True, use_fft=True)
    return dist.prob(x).numpy()
  
methods = pd.DataFrame({
    'identifier': [
        'basic',
        'seaborn',
        'KDEpy',
        'basicTF',
        'tfp',
        'tfpM',
        'tfpMB',
        'tfpMFFT'
    ],
    'label': [
        'Basic KDE with Python',
        'KDE using seaborn.distplot',
        'KDE using KDEpy.FFTKDE',
        'Basic KDE in TensorFlow',
        'KDE implemented as TensorFlow Probability Distribution Subclass',
        'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass',
        'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass with Binned Data',
        'KDE implemented as TensorFlow Probability MixtureSameFamily Subclass with Fast Fourier Transform'

    ],
    'function':[
        kde_basic,
        kde_seaborn,
        kde_kdepy_fft,
        kde_basic_tf,
        kde_tfp,
        kde_tfp_mixture,
        kde_tfp_mixture_with_binned_data,
        kde_tfp_mixture_with_fft
    ]
})
methods.set_index('identifier', drop=False, inplace=True)

estimations = pd.DataFrame()
estimations['x'] = np.linspace(-7.0, 7.0, num=n_testpoints, dtype=np.float32)

methods['runtime'] = np.NaN
for index, method in methods.iterrows():
    with Timer('Benchmarking') as timer:
        estimations[method['identifier']] = method['function'](data, estimations['x'])
        timer.stop()
    methods.at[method['identifier'], 'runtime'] = timer.elapsed

methods.drop('function', axis=1, inplace=True)

print(estimations)
print(methods)

ax = estimations.plot(x='x', y=['basic', 'KDEpy', 'basicTF', 'tfp', 'tfpM', 'tfpMB', 'tfpMFFT'], style=['-', '--', '-.', ':', '--', '-.', '--'])
plt.show()