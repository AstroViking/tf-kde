import tensorflow as tf
import numpy as np
from zfit import ztypes
from zfit_benchmark.timer import Timer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import inspect
from decimal import Decimal
from scipy.interpolate import UnivariateSpline

from tf_kde.benchmark import distributions as available_distributions
from tf_kde.benchmark import methods as available_methods

sns.set()
sns.set_context("paper")
plt.rc('axes', titlesize=8) 
plt.rc('axes', labelsize=6)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6) 
plt.rc('legend', fontsize=6)
plt.rc('figure', titlesize=8)

colormap = plt.get_cmap('gist_rainbow')

method_count = 0
method_colors = {
    'actual': colormap(0)
}

for method in inspect.getmembers(available_methods, inspect.isclass): method_count += 1
method_index = 1
for method, cls in inspect.getmembers(available_methods, inspect.isclass):
    method_colors[method] = colormap(1.*method_index/method_count)
    method_index += 1

def get_silverman_bandwidth(n, d=1): 
    return (n * (d + 2) / 4.)**(-1. / (d + 4))

def run_time_benchmark(methods, distributions, n_samples_list, n_testpoints, random_seed, xlim=[-10.0, 10.0], n_runs = 1, n_compilation_runs = 3):

    steps = ['instantiation', 'pdf', 'total']
    runtimes = pd.DataFrame(index=pd.MultiIndex.from_product([distributions, n_samples_list], names=['distribution', 'n_samples']), columns=pd.MultiIndex.from_product([steps, methods], names=['step', 'method']))
    runtimes = runtimes.sort_index()

    x = tf.linspace(tf.cast(xlim[0], ztypes.float), tf.cast(xlim[1], ztypes.float), num=n_testpoints)

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

    x = tf.linspace(tf.cast(xlim[0], ztypes.float), tf.cast(xlim[1], ztypes.float), num=n_testpoints)

    y = ['actual']
    y.extend(methods)

    estimations = pd.DataFrame(index=pd.MultiIndex.from_product([distributions, n_samples_list, x.numpy()], names=['distribution', 'n_samples', 'x']), columns=pd.Index(y, name='y'))
    estimations = estimations.sort_index()

    for method in methods:
        if hasattr(available_methods, method):

            method_to_instantiate = getattr(available_methods, method)

            for distribution in distributions:
                if hasattr(available_distributions, distribution):

                    distribution_object = getattr(available_distributions, distribution)
                    y_actual = distribution_object.prob(tf.cast(x, tf.float32)).numpy()

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


def plot_runtime(runtimes, distribution, methods, step, axes=None):

    if axes is None:
        figure, axes = plt.subplots()
    else:
        figure = axes.figure

    runtime = runtimes.xs(distribution).xs(step, axis=1)
    runtime.astype(np.float64).plot(kind='line', y=methods, color=method_colors, ax=axes, logy=True, logx=True, title=distribution)

    axes.set_xlabel('Number of samples ($n$)')
    axes.set_ylabel(f'{step.capitalize()} runtime [s]')
    axes.legend().set_title(None)

    return figure, axes


def plot_estimation(estimations, distribution, methods, n_samples_to_show, axes=None):

    if axes is None:
        figure, axes = plt.subplots()
    else:
        figure = axes.figure

    methods_to_show = ['actual']
    methods_to_show.extend(methods)

    estimation = estimations.xs((distribution, n_samples_to_show))
    estimation.astype(np.float64).plot(kind='line', y=methods_to_show, color=method_colors, ax=axes, title=distribution)

    integrated_square_errors = calculate_integrated_square_errors(estimation, methods)
    handles, labels = axes.get_legend_handles_labels()

    for key, label in enumerate(labels):
        if label != 'actual':
            labels[key] = label + f' ($ISE$: {integrated_square_errors[label]:.3e})'

    axes.legend(handles, labels)
    axes.set_xlabel('$x$')
    axes.set_ylabel('$P(x)$')

    return figure, axes


def plot_integrated_square_error(estimations, distribution, methods, axes=None):

    if axes is None:
        figure, axes = plt.subplots()
    else:
        figure = axes.figure

    estimation = estimations.xs(distribution)

    integrated_square_errors_list = {}
    for n_samples, specific_estimation in estimation.groupby('n_samples'):
        integrated_square_errors_list[n_samples] = calculate_integrated_square_errors(specific_estimation.xs(n_samples), methods)

    integrated_square_errors = pd.DataFrame.from_dict(integrated_square_errors_list, orient='index')
    integrated_square_errors.astype(np.float64).plot(kind='line', y=methods, color=method_colors, ax=axes, logy=True, logx=True, title=distribution)

    axes.set_xlabel('Number of samples ($n$)')
    axes.set_ylabel('$ISE$')

    return figure, axes


def plot_bar_integrated_square_error(estimations, distribution, methods, n_samples_to_show, axes=None):

    if axes is None:
        figure, axes = plt.subplots()
    else:
        figure = axes.figure

    estimation = estimations.xs((distribution, n_samples_to_show))
    integrated_square_errors = calculate_integrated_square_errors(estimation, methods)

    axes.set_title(distribution)
    axes.bar(list(integrated_square_errors.keys()), list(integrated_square_errors.values()))
    axes.set_xlabel('Method')
    axes.set_ylabel('$ISE$')

    return figure, axes


def plot_distributions(distributions, xlim):
    x = np.linspace(xlim[0], xlim[1], num=1000, dtype=np.float64)

    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        distribution_object = getattr(available_distributions, distribution)
        y = distribution_object.prob(x).numpy()
        axes[k].plot(x, y, color=method_colors['actual'])
        axes[k].set_title(distribution)
        axes[k].set_xlabel('$x$')
        axes[k].set_ylabel('$P(x)$')
        k +=1

    figure.tight_layout()

    return figure, axes


def plot_runtimes(runtimes, distributions, methods, step):
    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        plot_runtime(runtimes, distribution, methods, step, axes[k])

        if k == 0:
            handles, labels = axes[k].get_legend_handles_labels()
        axes[k].get_legend().remove()

        k += 1

    figure.tight_layout()
    figure.legend(handles, labels, ncol=10, loc='lower right', bbox_transform=figure.transFigure, borderpad=1)
    figure.set_figheight(7)
    figure.set_figwidth(8)
    figure.subplots_adjust(bottom = 1.0/8.0)

    return figure, axes


def plot_estimations(estimations, distributions, n_samples_to_show, methods):
    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        plot_estimation(estimations, distribution, methods, n_samples_to_show, axes[k])

        if k == 0:
            handles, labels = axes[k].get_legend_handles_labels()

        axes[k].get_legend().remove()

        k += 1

    figure.tight_layout()
    figure.legend(handles, labels, ncol=10, loc='lower right', bbox_transform=figure.transFigure, borderpad=1)
    figure.set_figheight(7)
    figure.set_figwidth(8)
    figure.subplots_adjust(bottom = 1.0/8.0)

    return figure, axes


def plot_integrated_square_errors(estimations, distributions, methods):
    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        plot_integrated_square_error(estimations, distribution, methods, axes[k])

        if k == 0:
            handles, labels = axes[k].get_legend_handles_labels()

        axes[k].get_legend().remove()

        k += 1

    figure.tight_layout()
    figure.legend(handles, labels, ncol=10, loc='lower right', bbox_transform=figure.transFigure, borderpad=1)
    figure.set_figheight(7)
    figure.set_figwidth(8)
    figure.subplots_adjust(bottom = 1.0/8.0)

    return figure, axes


def plot_bar_integrated_square_errors(estimations, distributions, n_samples_to_show, methods):
    figure, axes = generate_subplots(len(distributions))

    k = 0
    for distribution in distributions:
        plot_bar_integrated_square_error(estimations, distribution, methods, n_samples_to_show, axes[k])

        k += 1

    figure.tight_layout()

    return figure, axes