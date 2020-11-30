import tensorflow as tf
import numpy as np
from zfit import ztypes
from tf_kde.custom_ops import hofmeyr_kde

def calculate_estimate(grid, grid_data, betas, bandwidth):

    alpha = tf.size(betas, ztypes.int) - 1
    n = tf.size(grid, ztypes.int)
    bandwidth = tf.constant(0.01, ztypes.float)

    l = tf.TensorArray(ztypes.float, size=alpha+1, clear_after_read=False)
    r = tf.TensorArray(ztypes.float, size=alpha+1, clear_after_read=False)

    estimate_contributions = tf.TensorArray(ztypes.float, size=alpha+1, clear_after_read=False)

    def calculate_estimate_contributions(k, l, r, estimate_contributions):

        l_k = tf.TensorArray(ztypes.float, size=n, clear_after_read=False)
        r_k = tf.TensorArray(ztypes.float, size=n, clear_after_read=False)

        l_k = l_k.write(0, tf.math.pow(tf.constant(-1.0, ztypes.float) * grid[0], tf.cast(k, ztypes.float)) * grid_data[0])
        r_k = r_k.write(n-1, tf.constant(0.0, ztypes.float))

        def calculate_l_and_r(i, l_k, r_k):
            l_k = l_k.write(i, tf.math.pow(tf.constant(-1.0, ztypes.float) * grid[i], tf.cast(k, ztypes.float)) * grid_data[i] + tf.math.exp((grid[i-1] - grid[i]) / bandwidth) * l_k.read(i-1))
            r_k = r_k.write(n-i-1, tf.math.exp((grid[n - i - 1] - grid[n-i]) / bandwidth) * (tf.math.pow(grid[n-i], tf.cast(k, ztypes.float)) * grid_data[n-i] + r_k.read(n-i)))

            return i + 1, l_k, r_k

        i = tf.constant(1, ztypes.int)
        i, l_k, r_k = tf.while_loop(lambda i, l_k, r_k: tf.less(i, n), calculate_l_and_r, [i, l_k, r_k])

        l = l.write(k, l_k.concat())
        r = r.write(k, r_k.concat())

        k_factorial = tf.exp(tf.math.lgamma(tf.cast(k+1, ztypes.float)))

        estimate_contribution_k = tf.TensorArray(ztypes.float, size=alpha+1, clear_after_read=False)

        def calculate_estimate_contribution_k(j, estimate_contribution_k):

            j_factorial = tf.math.exp(tf.math.lgamma(tf.cast(j+1, ztypes.float)))
            k_j_factorial = tf.math.exp(tf.math.lgamma(tf.cast(k-j+1, ztypes.float)))
            binomial_coefficent = k_factorial/j_factorial/k_j_factorial

            estimate_contribution_k = estimate_contribution_k.write(j, binomial_coefficent * (tf.math.pow(grid, tf.cast(k-j, ztypes.float)) * l.read(j) + tf.math.pow(tf.constant(-1.0, ztypes.float) * grid, tf.cast(k-j, ztypes.float)) * r.read(j)))

            return tf.add(j, 1), estimate_contribution_k
        
        j = tf.constant(0, ztypes.int)
        j, estimate_contribution_k = tf.while_loop(lambda j, estimate_contribution_k: tf.less(j, k+1), calculate_estimate_contribution_k, [j, estimate_contribution_k])

        estimate_contributions = estimate_contributions.write(k, betas[k]/tf.math.pow(bandwidth, tf.cast(k, ztypes.float)) * tf.math.reduce_sum(estimate_contribution_k.stack(), axis=0))

        return tf.add(k, 1), l, r, estimate_contributions

    k = tf.constant(0, ztypes.int)
    k, l, r, estimate_contributions = tf.while_loop(lambda k, l, r, estimate_contributions: tf.less(k, alpha+1), calculate_estimate_contributions, [k, l, r, estimate_contributions])
    estimations = tf.math.reduce_sum(estimate_contributions.stack(), axis=0)

    return estimations

def _calculate_estimate_numpy(x, y, betas, h):

    n = np.size(x)
    alpha = np.size(betas) - 1

    l = np.zeros((alpha+1,n))
    r = np.zeros((alpha+1,n))

    estimations = np.zeros(n)

    for k in range(0, alpha+1):
        l[k,0] = np.power(-1.0 * x[0], k) * y[0]

        for i in range(1, n):
            l[k, i] = np.power(-1.0 * x[i], k) * y[i] + np.exp((x[i-1] - x[i]) / h) * l[k,i - 1]
            r[k, n - i - 1] = np.exp((x[n - i - 1] - x[n - i]) / h)*(np.power(x[n - i], k) * y[n-i] + r[k,n - i])
        
    for k in range(0, alpha +1):
        coef = np.zeros(alpha+1)
        coef[0] = 1
        coef[k] = 1

        if k > 1: 
            num = 1
            for j in range(2,k+1):
                num *= j

            denom1 = 1.0
            denom2 = num / k
            for j in range(2, k+1):
                coef[j - 1] = num / denom1 / denom2
                denom1 *= j
                denom2 /= (k - j + 1)

        denom = np.power(h, k)

        for i in range(0, n):
            exp_mult = 1.0 #exp((x[ix - 1] - x_eval[i]) / h);

            estimate_contribution = 0.0
            for j in range(0,k+1):
                estimate_contribution += betas[k] * coef[j] * (np.power(x[i], k-j) * l[j, i] * exp_mult + np.power(-x[i], k-j) * r[j, i]) / denom
            estimations[i] += estimate_contribution

    return estimations

def calculate_estimate_numpy(grid, grid_data, betas, bandwidth):
    return tf.numpy_function(func=_calculate_estimate_numpy, inp=[grid, grid_data, betas, bandwidth], Tout=ztypes.float)

def calculate_estimate_cpp(grid, grid_data, betas, bandwidth):
    return hofmeyr_kde(grid, grid_data, grid, betas, bandwidth)
