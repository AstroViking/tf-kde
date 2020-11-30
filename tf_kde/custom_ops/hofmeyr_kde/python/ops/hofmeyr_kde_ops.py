"""Use hofmeyr_kde ops in python."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

hofmeyr_kde_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('../../cc/_hofmeyr_kde_ops.so'))
hofmeyr_kde = hofmeyr_kde_ops.hofmeyr_kde