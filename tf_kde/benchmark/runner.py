import tensorflow as tf
import numpy as np
import zfit as zfit
from KDEpy import NaiveKDE, FFTKDE
from tf_kde.distribution import KernelDensityEstimation, KernelDensityEstimationBasic, KernelDensityEstimationZfit
from zfit_benchmark.timer import Timer
import seaborn as sns
import pandas as pd
from decimal import Decimal
from scipy.interpolate import UnivariateSpline

from tf_kde.benchmark import distributions as available_distributions
from tf_kde.benchmark import methods as available_methods

def get_silverman_bandwidth(n, d=1): 
    return (n * (d + 2) / 4.)**(-1. / (d + 4))

def run_time_benchmark(methods, distributions, n_samples_list, n_runs, n_testpoints, random_seed, additional_run_to_initialize = True, xlim=[-5.0, 5.0]):

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
                    raise NameError('Distribution \'%s\' is not defined!')
        else:
            raise NameError('Method \'%s\' is not defined!')

    return runtimes


def run_error_benchmark(methods, distributions, n_samples_list, n_testpoints, random_seed, xlim=[-5.0, 5.0]):

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
                        #estimations.loc[(distribution, n_samples), method + '_error'] = y_actual - y_estimate

                else:
                    raise NameError('Distribution \'%s\' is not defined!')
        else:
            raise NameError('Method \'%s\' is not defined!')

    return estimations

if __name__ == "__main__":

    random_seed = 756454
    n_testpoints = 1024
    n_runs = 3
    methods_to_evaluate = [
        #'basic',
        'kdepy_fft',
        'zfit_binned',
        'zfit_fft',
        #'zfit_ffts'
    ]
    distributions_to_evaluate = [
        'gaussian',
        'uniform',
        'skewed_bimodal',
        'claw',
        'mix_3gauss_1exp_1uni'
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
        -7,
        7
    ]

    runtimes = run_time_benchmark(methods_to_evaluate, distributions_to_evaluate, n_samples_list, n_runs, n_testpoints, random_seed, True, xlim)
    estimations = run_error_benchmark(methods_to_evaluate, distributions_to_evaluate, n_samples_list, n_testpoints, random_seed, xlim)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
        
    n_samples_to_show = 1e7
    n_methods = len(methods_to_evaluate)
    n_distributions = len(distributions_to_evaluate)

    n_columns = 2
    n_rows = int(np.ceil(n_distributions / n_columns))

    f1, a1 = plt.subplots(n_rows, n_columns)
    f2, a2 = plt.subplots(n_rows, n_columns)

    a1 = a1.flatten()
    a2 = a2.flatten()

    methods_to_show = ['actual']
    methods_to_show.extend(methods_to_evaluate)

    k = 0
    for distribution in distributions_to_evaluate:

        distribution_runtimes = runtimes.xs(distribution)
        distribution_runtimes.astype(np.float64).plot(kind='line', y=methods_to_evaluate, ax=a1[k], logy=True,logx=True)
        a1[k].set_title(distribution)

        distribution_estimations = estimations.xs((distribution, n_samples_to_show))
        distribution_estimations.astype(np.float64).plot(kind='line', y=methods_to_show, ax=a2[k])
        a2[k].set_title(distribution)

        k += 1

        handles, labels = a2[k].get_legend_handles_labels()

        integrated_square_error = np.zeros(n_methods)

        j = 0

        new_labels = []
        for method in available_methods:
            square_error = (distribution_estimations.loc[:, label] - distribution_estimations.loc[:, 'actual'])**2
            spl = UnivariateSpline(distribution_estimations.index.to_numpy(), square_error)
            integrated_square_error[j] = spl.integral(xlim[0], xlim[1])

            new_labels.append(method + ' (ISE: ' + '{:3e}'.format(integrated_square_error[j]) + ')')

            j += 1

        print(integrated_square_error)

        a2[k].legend(handles, new_labels)
    
    plt.show()




