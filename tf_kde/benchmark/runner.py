import tensorflow as tf
import numpy as np
import zfit as zfit
from KDEpy import NaiveKDE, FFTKDE
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic, KernelDensityEstimationZfit
from zfit_benchmark.timer import Timer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from decimal import Decimal
from scipy.interpolate import UnivariateSpline

from tf_kde.benchmark import distributions as available_distributions
from tf_kde.benchmark import methods as available_methods

sns.set()
plt.rc('legend',fontsize=6)

def get_silverman_bandwidth(n, d=1): 
    return (n * (d + 2) / 4.)**(-1. / (d + 4))

def run_time_benchmark(methods, distributions, n_samples_list, n_runs, n_testpoints, random_seed, additional_run_to_initialize = True, xlim=[-10.0, 10.0]):

    runtimes = pd.DataFrame(index=pd.MultiIndex.from_product([distributions, n_samples_list], names=['distribution', 'n_samples']), columns=pd.Index(data=methods, name='method'))
    runtimes = runtimes.sort_index()

    x = np.linspace(xlim[0], xlim[1], num=n_testpoints, dtype=np.float32)

    for method in methods:
        if hasattr(available_methods, method):

            method_to_call = getattr(available_methods, method)

            for distribution in distributions:

                if hasattr(available_distributions, distribution):

                    for n_samples in n_samples_list:

                        bandwidth = get_silverman_bandwidth(n_samples)

                        data = getattr(available_distributions, distribution).sample(sample_shape=n_samples, seed=random_seed).numpy()

                        time = Decimal(0.0)

                        if additional_run_to_initialize:
                            method_to_call(data, x, bandwidth)

                        for k in range(n_runs):
                            with Timer('Benchmarking') as timer:
                                method_to_call(data, x, bandwidth)
                                timer.stop()
                            time += timer.elapsed
                        
                        runtimes.at[(distribution, n_samples), method] = time / n_runs

                else:
                    raise NameError(f'Distribution \'{distribution}\' is not defined!')
        else:
            raise NameError(f'Method \'{method}\' is not defined!')

    return runtimes


def run_error_benchmark(methods, distributions, n_samples_list, n_testpoints, random_seed, xlim=[-10.0, 10.0]):

    x = np.linspace(xlim[0], xlim[1], num=n_testpoints, dtype=np.float32)

    y = ['actual']
    y.extend(methods)

    estimations = pd.DataFrame(index=pd.MultiIndex.from_product([distributions, n_samples_list, x], names=['distribution', 'n_samples', 'x']), columns=pd.Index(y, name='y'))
    estimations = estimations.sort_index()

    for method in methods:
        if hasattr(available_methods, method):

            method_to_call = getattr(available_methods, method)

            for distribution in distributions:
                if hasattr(available_distributions, distribution):

                    distribution_object = getattr(available_distributions, distribution)
                    y_actual = distribution_object.prob(x).numpy()

                    for n_samples in n_samples_list:

                        estimations.loc[(distribution, n_samples), 'actual'] = y_actual

                        bandwidth = get_silverman_bandwidth(n_samples)

                        data = distribution_object.sample(sample_shape=n_samples, seed=random_seed).numpy()
                        y_estimate = method_to_call(data, x, bandwidth)
 
                        estimations.loc[(distribution, n_samples), method] = y_estimate

                else:
                    raise NameError(f'Distribution \'{distribution}\' is not defined!')
        else:
            raise NameError(f'Method \'{method}\' is not defined!')

    return estimations


def calculate_integrated_square_errors(estimation, methods):

    integrated_square_errors = {}

    for method in methods:
        square_error = (estimation.loc[:, method] - estimation.loc[:, 'actual'])**2
        spline = UnivariateSpline(estimation.index.to_numpy(), square_error)
        integrated_square_errors[method] = spline.integral(xlim[0], xlim[1])
    
    return integrated_square_errors


def generate_subplots(n_distributions, n_columns = 2):

    n_rows = int(np.ceil(n_distributions / n_columns))

    figure, axes = plt.subplots(n_rows, n_columns)
    axes = axes.flatten()

    return figure, axes


def plot_runtime(runtimes, distribution, methods, axes):
    runtime = runtimes.xs(distribution)
    runtime.astype(np.float64).plot(kind='line', y=methods, ax=axes, logy=True,logx=True, title=distribution)


def plot_estimation(estimations, distribution, methods, n_samples_to_show, axes):

    methods_to_show = ['actual']
    methods_to_show.extend(methods)

    estimation = estimations.xs((distribution, n_samples_to_show))
    estimation.astype(np.float64).plot(kind='line', y=methods_to_show, ax=axes, title=distribution)

    integrated_square_errors = calculate_integrated_square_errors(estimation, methods)
    handles, labels = axes.get_legend_handles_labels()

    for key, label in enumerate(labels):
        if label != 'actual':
            labels[key] = label + f' (ISE: {integrated_square_errors[label]:.3e})'

    axes.legend(handles, labels)


def plot_distributions(distributions, xlim, n_columns):
    x = np.linspace(xlim[0], xlim[1], num=1000, dtype=np.float64)

    subplots = generate_subplots(len(distributions), n_columns)

    k = 0
    for distribution in distributions:
        distribution_object = getattr(available_distributions, distribution)
        y = distribution_object.prob(x).numpy()
        subplots[k].plot(x, y)
        k +=1

    distribution_object.prob(x).numpy()


def plot_runtimes(runtimes, distributions, methods):
    figure, axes = generate_subplots(len(distributions_to_evaluate))

    k = 0
    for distribution in distributions:
        plot_runtime(runtimes, distribution, methods, axes[k])
        k += 1

    return figure, axes


def plot_estimations(estimations, distributions, n_samples_to_show, methods):
    figure, axes = generate_subplots(len(distributions_to_evaluate))

    k = 0
    for distribution in distributions:
        plot_estimation(estimations, distribution, methods, n_samples_to_show, axes[k])
        k += 1

    return figure, axes


if __name__ == "__main__":

    random_seed = 756454
    n_testpoints = 1024
    n_runs = 10
    methods_to_evaluate = [
        #'basic',
        'kdepy_fft',
        #'kdepy_fft_isj',
        'zfit_binned',
        #'zfit_simple_binned',
        'zfit_fft',
        #'zfit_ffts',
        'zfit_fft_with_isj_bandwidth',
        'zfit_isj',
        'zfit_adaptive'
    ]
    distributions_to_evaluate = [
        'gaussian',
        'uniform',
        'bimodal',
        'mix_3gauss_1exp_1uni',
        'claw',
        'asymmetric_double_claw'
    ]

    n_samples_list = [
        1e1,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7
    ]

    xlim = [
        -10,
        10
    ]

    runtimes = run_time_benchmark(methods_to_evaluate, distributions_to_evaluate, n_samples_list, n_runs, n_testpoints, random_seed, True, xlim)
    estimations = run_error_benchmark(methods_to_evaluate, distributions_to_evaluate, n_samples_list, n_testpoints, random_seed, xlim)
        
    n_samples_to_show = 1e7

    plot_runtimes(runtimes, distributions_to_evaluate, methods_to_evaluate)
    plot_estimations(estimations, distributions_to_evaluate, n_samples_to_show, methods_to_evaluate)
    plt.show()