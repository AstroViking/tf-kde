import tensorflow as tf
import numpy as np
from tf_quant_finance.math import root_search
from zfit import ztypes

from tf_kde.helper import binning as binning_helper

def _fixed_point(t, N, squared_integers, grid_data_dct2):
    r"""
    Compute the fixed point as described in the paper by Botev et al.
    .. math:
        t = \xi \gamma^{5}(t)
    Parameters
    ----------
    t : float
        Initial guess.
    N : int
        Number of data points.
    squared_integers : array-like
        The numbers [1, 2, 9, 16, ...]
    grid_data_dct2 : array-like
        The DCT of the original data, divided by 2 and squared.
    Examples
    --------
    >>> # From the matlab code
    >>> ans = _fixed_point(0.01, 50, np.arange(1, 51), np.arange(1, 51))
    >>> assert np.allclose(ans, 0.0099076220293967618515)
    >>> # another
    >>> ans = _fixed_point(0.07, 25, np.arange(1, 11), np.arange(1, 11))
    >>> assert np.allclose(ans, 0.068342291525717486795)
    References
    ----------
     - Implementation by Daniel B. Smith, PhD, found at
       https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py
    """

    # This is important, as the powers might overflow if not done
    #squared_integers = np.asfarray(squared_integers, dtype=FLOAT)
    #grid_data_dct2 = np.asfarray(grid_data_dct2, dtype=FLOAT)

    # ell = 7 corresponds to the 5 steps recommended in the paper
    ell = tf.constant(7, ztypes.float)

    # Fast evaluation of |f^l|^2 using the DCT, see Plancherel theorem
    f = tf.constant(0.5, ztypes.float) * tf.math.pow(tf.constant(np.pi, ztypes.float), (tf.constant(2.0, ztypes.float) * ell)) * tf.math.reduce_sum(tf.math.pow(squared_integers, ell) * grid_data_dct2 * tf.math.exp(-squared_integers * tf.math.pow(tf.constant(np.pi, ztypes.float), tf.constant(2.0, ztypes.float)) * t)) 

    def calc_f(s, f):
        s = tf.constant(s, ztypes.float)

        # Step one: estimate t_s from |f^(s+1)|^2
        odd_numbers_prod = tf.math.reduce_prod(tf.range(tf.constant(1.0, ztypes.float), tf.constant(2.0, ztypes.float) * s + tf.constant(1, ztypes.float), 2))
        K0 = odd_numbers_prod / tf.math.sqrt(tf.constant(2.0 * np.pi, ztypes.float))
        const = (tf.constant(1.0, ztypes.float) + tf.math.pow(tf.constant(1.0 / 2.0, ztypes.float), s + tf.constant(1.0 / 2.0, ztypes.float))) / tf.constant(3.0, ztypes.float)
        time = tf.math.pow(tf.constant(2.0, ztypes.float) * const * K0 / (N * f), tf.constant(2.0, ztypes.float) / (tf.constant(3.0, ztypes.float) + tf.constant(2.0, ztypes.float) * s))

        # Step two: estimate |f^s| from t_s
        f = tf.constant(0.5, ztypes.float) * tf.math.pow(tf.constant(np.pi, ztypes.float), (tf.constant(2.0, ztypes.float) * s)) * tf.math.reduce_sum(tf.math.pow(squared_integers, s) * grid_data_dct2 * tf.math.exp(-squared_integers * tf.math.pow(tf.constant(np.pi, ztypes.float), tf.constant(2.0, ztypes.float)) * time))

        return f

    # Maybe we can do this more dynamically without losing performance?!
    # s = tf.reverse(tf.range(2, ell))
    f = calc_f(6, f)
    f = calc_f(5, f)
    f = calc_f(4, f)
    f = calc_f(3, f)
    f = calc_f(2, f)

    # This is the minimizer of the AMISE
    t_opt = tf.math.pow(tf.constant(2 *  np.sqrt(np.pi), ztypes.float) * N * f, tf.constant(-2.0 / 5.0, ztypes.float))

    # Return the difference between the original t and the optimal value
    return t - t_opt


def _find_root(function, N, squared_integers, grid_data_dct2):
    """
    Root finding algorithm. Based on MATLAB implementation by Botev et al.
    >>> # From the matlab code
    >>> ints = np.arange(1, 51)
    >>> ans = _root(_fixed_point, N=50, args=(50, ints, ints))
    >>> np.allclose(ans, 9.237610787616029e-05)
    True
    """
    # From the implementation by Botev, the original paper author
    # Rule of thumb of obtaining a feasible solution
    N2 = tf.math.maximum(tf.math.minimum(tf.constant(1050, ztypes.float), N), tf.constant(50, ztypes.float))
    tol = 10e-12 + 0.01 * (N2 - 50) / 1000
    left_bracket = tf.constant(0.0, dtype=ztypes.float)
    right_bracket = tf.constant(10e-12, ztypes.float) + tf.constant(0.01, ztypes.float) * (N2 - tf.constant(50, ztypes.float)) / tf.constant(1000, ztypes.float)

    converged = tf.constant(False)
    t_star = tf.constant(0.0, dtype=ztypes.float)


    def fixed_point_function(t): 
        return _fixed_point(t, N, squared_integers, grid_data_dct2)

    condition = lambda right_bracket, converged, t_star: tf.math.logical_not(converged)
    
    def body(right_bracket, converged, t_star): 

        t_star, value_at_t_star, num_iterations, converged = root_search.brentq(
            fixed_point_function,
            left_bracket,
            right_bracket,
            None,
            None,
            2e-12
        )

        # Why does this give the right answer???
        t_star = t_star - value_at_t_star

        right_bracket = right_bracket * tf.constant(2.0, ztypes.float)

        return right_bracket, converged, t_star

    # While a solution is not found, increase the tolerance and try again
    right_bracket, converged, t_star = tf.while_loop(condition, body, [right_bracket, converged, t_star])

    return t_star


def improved_sheather_jones(data):
    
    n = 2 ** 10
    # Setting `percentile` higher decreases the chance of overflow
    grid = binning_helper.generate_grid(data, n, 6.0, 0.5)

    # Create an equidistant grid
    R = tf.cast(tf.reduce_max(data) - tf.reduce_min(data), ztypes.float)

    dx = R / tf.constant(n - 1, ztypes.float)
    data_unique, data_unique_indexes = tf.unique(data)
    N = tf.cast(tf.size(data_unique), ztypes.float)

    # Use linear binning to bin the data on an equidistant grid, this is a
    # prerequisite for using the FFT (evenly spaced samples)
    grid_data = binning_helper.bin_linear(data, grid)

    # Compute the type 2 Discrete Cosine Transform (DCT) of the data
    grid_data_dct = tf.signal.dct(grid_data, type=2)

    # Compute the bandwidth
    squared_integers = tf.math.pow(tf.range(1, n, dtype=ztypes.float), tf.constant(2, ztypes.float))
    grid_data_dct2 = tf.math.pow(grid_data_dct[1:], 2) / 4

    # Solve for the optimal (in the AMISE sense) t
    t_star = _find_root(_fixed_point, N, squared_integers, grid_data_dct2)

    # The remainder of the algorithm computes the actual density
    # estimate, but this function is only used to compute the
    # bandwidth, since the bandwidth may be used for other kernels
    # apart from the Gaussian kernel

    # Smooth the initial data using the computed optimal t
    # Multiplication in frequency domain is convolution
    # integers = np.arange(n, dtype=np.float)
    # grid_data_dct2_t = grid_data_dct * np.exp(-integers**2 * np.pi ** 2 * t_star / 2)

    # Diving by 2 done because of the implementation of fftpack.idct
    # density = fftpack.idct(grid_data_dct2_t) / (2 * R)

    # Due to overflow, some values might be smaller than zero, correct it
    # density[density < 0] = 0.
    bandwidth = tf.math.sqrt(t_star) * R
    return bandwidth

if __name__ == "__main__":  

    from KDEpy import bw_selection
    import tensorflow_probability as tfp
    import tensorflow_probability.python.distributions as tfd

    random_seed = 1978239485
    n_datapoints = 1e7

    from tf_kde.benchmark import distributions as available_distributions

    data  = available_distributions.bimodal.sample(n_datapoints, seed=random_seed).numpy().astype(np.float64)

    tf_bw = improved_sheather_jones(data.astype(np.float64)).numpy()
    kdepy_bw = bw_selection.improved_sheather_jones(data.astype(np.float64).reshape(len(data), 1))

    print(tf_bw)
    print(kdepy_bw)
    print((n_datapoints * (1 + 2) / 4.)**(-1. / (1 + 4)))