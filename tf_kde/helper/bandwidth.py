from tf_kde.helper import improved_sheather_jones as isj_helper

def improved_sheather_jones(data, num_grid_points=1024, binning_method = 'linear', weights=None):
    return isj_helper.calculate_bandwidth(data)