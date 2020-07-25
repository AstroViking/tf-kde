import tensorflow as tf
from zfit import ztypes
from tf_quant_finance.math import root_search

def find_practical_support_bandwidth(kernel, bandwidth, absolute_tolerance=10e-5):
    """
    Return the support for practical purposes. Used to find a support value
    for computations for kernel functions without finite (bounded) support.
    """
    absolute_root_tolerance = 1e-3
    relative_root_tolerance = root_search.default_relative_root_tolerance(ztypes.float)
    function_tolerance = 0
    
    kernel_instance = kernel(loc=0, scale=bandwidth)

    def objective_fn(x): 
        return kernel_instance.prob(x) - tf.constant(absolute_tolerance, ztypes.float)

    roots, value_at_roots, num_iterations, converged = root_search.brentq(
        objective_fn,
        tf.constant(0.0, dtype=ztypes.float),
        tf.constant(8.0, dtype=ztypes.float) * bandwidth,
        absolute_root_tolerance=absolute_root_tolerance,
        relative_root_tolerance=relative_root_tolerance,
        function_tolerance=function_tolerance
    )

    return roots + absolute_root_tolerance