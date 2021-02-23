from shapely.geometry import MultiPoint
import math
import numpy as np


def globalmercator_bounds_2_shapely_polygon(bounds):
    points = [(bounds[1], bounds[0]), (bounds[3], bounds[2])]
    return MultiPoint(points).envelope


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def tile_fname_to_x_y_zoom(fname):
    name_only = fname.split(".")[0]
    fname_parts = name_only.split("_")
    return int(fname_parts[0]), int(fname_parts[1]), int(fname_parts[2])


def make_uint8_mask(image):
    mask = np.where(image == 0, 0, 255)
    return np.where(np.sum(mask, axis=2) == 0, 0, 255).astype(np.uint8)
