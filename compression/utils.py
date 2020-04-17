from functools import *

from pypcd.pypcd import PointCloud

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def load_pointcloud(path):
    return PointCloud.from_path(path)
