import tensorflow as tf
from zfit import ztypes

def generate_grid(data, num_grid_points): 
    minimum = tf.math.reduce_min(data)
    maximum = tf.math.reduce_max(data)
    return tf.linspace(minimum, maximum, num=num_grid_points)

def bin(binning_method, data, grid, weights):

    if binning_method == 'simple':
        return bin_simple(data, grid, weights)
    else:
        return bin_linear(data, grid, weights)

def bin_simple(data, grid, weights):

    minimum = tf.math.reduce_min(grid)
    maximum = tf.math.reduce_max(grid)
    bin_count = tf.size(grid)

    bincount = tf.histogram_fixed_width(data, [minimum, maximum], bin_count)

    #!TODO include weights in calculation

    return bincount


def bin_linear(data, grid, weights):

    if weights is None:
        weights = tf.ones_like(data, ztypes.float)            

    weights = weights / tf.reduce_sum(weights)

    grid_size = tf.size(grid)
    grid_min = tf.math.reduce_min(grid)
    grid_max = tf.math.reduce_max(grid)
    num_intervals = tf.math.subtract(grid_size, tf.constant(1))
    dx = tf.math.divide(tf.math.subtract(grid_max, grid_min), tf.cast(num_intervals, ztypes.float))

    transformed_data = tf.math.divide(tf.math.subtract(data, grid_min), dx)

    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    integral = tf.math.floor(transformed_data)
    fractional = tf.math.subtract(transformed_data, integral)
    
    # Compute the weights for left and right side of the linear binning routine
    frac_weights = tf.math.multiply(fractional, weights)
    neg_frac_weights = tf.math.subtract(weights, frac_weights)

    #tf.math.bincount only works with tf.int32
    bincount_left = tf.roll(tf.concat(tf.math.bincount(tf.cast(integral, tf.int32), weights=frac_weights, minlength=grid_size, maxlength=grid_size), tf.constant(0)), shift=1, axis=0)
    bincount_right = tf.math.bincount(tf.cast(integral, tf.int32), weights=neg_frac_weights, minlength=grid_size, maxlength=grid_size)

    bincount = tf.cast(tf.add(bincount_left, bincount_right), ztypes.float)

    return bincount