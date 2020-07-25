import tensorflow as tf
from zfit import ztypes

import tf_kde.helper.support as support_helper

def convolve_data_with_kernel(kernel, bandwidth, data, grid, fft_method = 'conv1d', support = None):

        kernel_grid_min = tf.math.reduce_min(grid)
        kernel_grid_max = tf.math.reduce_max(grid)

        num_grid_points = tf.size(grid, ztypes.int)
        num_intervals = num_grid_points - tf.constant(1, ztypes.int)
        space_width = kernel_grid_max - kernel_grid_min
        dx = space_width / tf.cast(num_intervals, ztypes.float)
            
        L = tf.cast(num_grid_points, ztypes.float)

        if support is not None:
            support_bandwidth = support * bandwidth
        else:
            support_bandwidth = support_helper.find_practical_support_bandwidth(kernel, bandwidth)

        L = tf.minimum(tf.math.floor(support_bandwidth / dx),  L)

        # Calculate the kernel weights
        kernel_grid = tf.linspace(tf.constant(0, ztypes.float), dx * L, tf.cast(L, ztypes.int) + tf.constant(1, ztypes.int))
        kernel_weights = kernel(loc=0, scale=bandwidth).prob(kernel_grid)
        kernel_weights = tf.concat(values=[tf.reverse(kernel_weights, axis = [0])[:-1], kernel_weights], axis=0)

        c = data
        k = kernel_weights

        if fft_method  == 'conv1d':
            c_size = tf.size(c, ztypes.int)
            c = tf.reshape(c, [1, c_size, 1], name='c')
            
            k_size = tf.size(k, ztypes.int)
            k = tf.reshape(k, [k_size, 1, 1], name='k')

            return tf.squeeze(tf.nn.conv1d(c, k, 1, 'SAME')) 
        
        else:     

            P = tf.math.pow(tf.constant(2, ztypes.int), tf.cast(tf.math.ceil(tf.math.log(tf.constant(3.0, ztypes.float) * tf.cast(num_grid_points, ztypes.float) - tf.constant(1.0, ztypes.float)) / tf.math.log(tf.constant(2.0, ztypes.float))), ztypes.int))

            right_padding = tf.cast(P, ztypes.int) - tf.constant(2, ztypes.int) * num_grid_points - tf.constant(1, ztypes.int)
            left_padding = num_grid_points - tf.constant(1, ztypes.int)

            c = tf.pad(data, [[left_padding, right_padding]])
            k = tf.pad(kernel_weights, [[0, right_padding]])

            result = tf.signal.irfft(tf.signal.rfft(c) * tf.signal.rfft(k))
            start, end = tf.constant(2, ztypes.int) * num_grid_points - tf.constant(1, ztypes.int), tf.constant(3, ztypes.int) * num_grid_points - tf.constant(1, ztypes.int)
            result = result[start:end]

            return result