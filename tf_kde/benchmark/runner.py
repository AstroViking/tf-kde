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
plt.rc('axes', titlesize=8) 
plt.rc('axes', labelsize=6)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6) 
plt.rc('legend', fontsize=6)
plt.rc('figure', titlesize=8)

def get_silverman_bandwidth(n, d=1): 
    return (n * (d + 2) / 4.)**(-1. / (d + 4))

def run_time_benchmark(methods, distributions, n_samples_list, n_testpoints, random_seed, xlim=[-10.0, 10.0], n_runs = 1, n_compilation_runs = 3):

    steps = ['instantiation', 'pdf', 'total']
    runtimes = pd.DataFrame(index=pd.MultiIndex.from_product([distributions, n_samples_list], names=['distribution', 'n_samples']), columns=pd.MultiIndex.from_product([steps, methods], names=['step', 'method']))
    runtimes = runtimes.sort_index()

    x = np.linspace(xlim[0], xlim[1], num=n_testpoints, dtype=np.float32)

    for method in methods:
        if hasattr(available_methods, method):
            
            method_to_instantiate = getattr(available_methods, method)

            for distribution in distributions:

                if hasattr(available_distributions, distribution):

                    for n_samples in n_samples_list:

                        bandwidth = get_silverman_bandwidth(n_samples)
                        data = getattr(available_distributions, distribution).sample(sample_shape=n_samples, seed=random_seed).numpy()

                        with Timer('Instantiation') as instantiation_timer:
                            kde = method_to_instantiate(data, bandwidth, xlim)
                            instantiation_timer.stop()

                        for k in range(n_compilation_runs):
                            kde.pdf(x)

                        pdf_time = Decimal(0.0)
                        for k in range(n_runs):
                            with Timer('pdf') as pdf_timer:
                                kde.pdf(x)
                                pdf_timer.stop()  
                            pdf_time += pdf_timer.elapsed
                        pdf_time /= n_runs
                        
                        runtimes.at[(distribution, n_samples), ('instantiation', method)] = instantiation_timer.elapsed
                        runtimes.at[(distribution, n_samples), ('pdf', method)] = pdf_time
                        runtimes.at[(distribution, n_samples), ('total', method)] = instantiation_timer.elapsed + pdf_time

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

            method_to_instantiate = getattr(available_methods, method)

            for distribution in distributions:
                if hasattr(available_distributions, distribution):

                    distribution_object = getattr(available_distributions, distribution)
                    y_actual = distribution_object.prob(x).numpy()

                    for n_samples in n_samples_list:

                        estimations.loc[(distribution, n_samples), 'actual'] = y_actual

                        bandwidth = get_silverman_bandwidth(n_samples)

                        data = distribution_object.sample(sample_shape=n_samples, seed=random_seed).numpy()

                        kde = method_to_instantiate(data, bandwidth, xlim)
                        y_estimate = kde.pdf(x)
 
                        estimations.loc[(distribution, n_samples), method] = y_estimate

                else:
                    raise NameError(f'Distribution \'{distribution}\' is not defined!')
        else:
            raise NameError(f'Method \'{method}\' is not defined!')

    return estimations


def calculate_integrated_square_errors(estimation, methods):

    integrated_square_errors = {}

    for method in methods:
        x = estimation.index.to_numpy()
        square_error = (estimation.loc[:, method] - estimation.loc[:, 'actual'])**2
        spline = UnivariateSpline(x, square_error)
        integrated_square_errors[method] = spline.integral(np.min(x), np.max(x))
    
    return integrated_square_errors


def generate_subplots(n_distributions, n_columns = 2):

    n_rows = int(np.ceil(n_distributions / n_columns))

    figure, axes = plt.subplots(n_rows, n_columns)
    axes = axes.flatten()

    return figure, axes


def plot_runtime(runtimes, distribution, methods, step, axes):

    runtime = runtimes.xs(distribution).xs(step, axis=1)
    runtime.astype(np.float64).plot(kind='line', y=methods, ax=axes, logy=True,logx=True, title=distribution)

    axes.set_xlabel('Number of samples')
    axes.set_ylabel(f'{step.capitalize()} runtime [s]')
    axes.legend().set_title(None)


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
    axes.set_xlabel('x')
    axes.set_ylabel('P(x)')


def plot_distributions(distributions, xlim):
    x = np.linspace(xlim[0], xlim[1], num=1000, dtype=np.float64)

    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        distribution_object = getattr(available_distributions, distribution)
        y = distribution_object.prob(x).numpy()
        axes[k].plot(x, y)
        axes[k].set_title(distribution)
        axes[k].set_xlabel('x')
        axes[k].set_ylabel('P(x)')
        k +=1

    figure.tight_layout()

    return figure, axes


def plot_runtimes(runtimes, distributions, methods, step):
    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        plot_runtime(runtimes, distribution, methods, step, axes[k])
        k += 1

    figure.tight_layout()

    return figure, axes


def plot_estimations(estimations, distributions, n_samples_to_show, methods):
    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        plot_estimation(estimations, distribution, methods, n_samples_to_show, axes[k])
        k += 1

    figure.tight_layout()

    return figure, axes