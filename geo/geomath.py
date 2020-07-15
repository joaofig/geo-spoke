import numpy as np
import math
# from numba import jit, prange


def arr_haversine(loc1: np.ndarray,
                  loc2: np.ndarray) -> np.ndarray:
    earth_radius = 6378137.0

    dim1 = loc1.shape[0]
    dim2 = loc2.shape[0]

    rad_loc1 = np.radians(loc1)
    rad_loc2 = np.radians(loc2)

    meters = np.zeros((dim1, dim2))

    for i in range(dim2):
        d_lon = rad_loc1[:, 1].ravel() - rad_loc2[i, 1]
        d_lat = rad_loc1[:, 0].ravel() - rad_loc2[i, 0]

        a = np.sin(d_lat / 2.0) ** 2 + np.cos(rad_loc1[:, 0]) * np.cos(rad_loc2[i, 0]) \
            * np.sin(d_lon / 2.0) ** 2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        meters[:, i] = earth_radius * c
    return meters
#
#
# @jit(nopython=True, parallel=True)
# def numba_haversine(lat1: np.ndarray,
#                     lon1: np.ndarray,
#                     lat2: np.ndarray,
#                     lon2: np.ndarray) -> np.ndarray:
#
#     meters = np.zeros((lat1.shape[0], lat2.shape[0]))
#     earth_radius = 6378137.0
#
#     rad_lat1 = np.radians(lat1)
#     rad_lon1 = np.radians(lon1)
#     rad_lat2 = np.radians(lat2)
#     rad_lon2 = np.radians(lon2)
#
#     for i in prange(lat2.shape[0]):
#         d_lon = rad_lon2[i] - rad_lon1
#         d_lat = rad_lat2[i] - rad_lat1
#
#         a = np.sin(d_lat/2.0)**2 + np.cos(rad_lat1) * np.cos(rad_lat2[i]) \
#             * np.sin(d_lon/2.0)**2
#
#         c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
#         meters[:, i] = earth_radius * c
#     return meters


def vec_haversine(lat1: np.ndarray,
                  lon1: np.ndarray,
                  lat2: np.ndarray,
                  lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance calculation
    :param lat1: Array of initial latitudes in degrees
    :param lon1: Array of initial longitudes in degrees
    :param lat2: Array of destination latitudes in degrees
    :param lon2: Array of destination longitudes in degrees
    :return: Array of distances in meters
    """
    earth_radius = 6378137.0

    rad_lat1 = np.radians(lat1)
    rad_lon1 = np.radians(lon1)
    rad_lat2 = np.radians(lat2)
    rad_lon2 = np.radians(lon2)

    d_lon = rad_lon2 - rad_lon1
    d_lat = rad_lat2 - rad_lat1

    a = np.sin(d_lat/2.0)**2 + np.cos(rad_lat1) * np.cos(rad_lat2) \
        * np.sin(d_lon/2.0)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = earth_radius * c
    return meters


def num_haversine(lat1: float,
                  lon1: float,
                  lat2: float,
                  lon2: float) -> float:
    """
    Vectorized haversine distance calculation
    :param lat1: Initial latitude in degrees
    :param lon1: Initial longitude in degrees
    :param lat2: Destination latitude in degrees
    :param lon2: Destination longitude in degrees
    :return: Distances in meters
    """
    earth_radius = 6378137.0

    rad_lat1 = math.radians(lat1)
    rad_lon1 = math.radians(lon1)
    rad_lat2 = math.radians(lat2)
    rad_lon2 = math.radians(lon2)

    d_lon = rad_lon2 - rad_lon1
    d_lat = rad_lat2 - rad_lat1

    a = math.sin(d_lat/2.0)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) \
        * math.sin(d_lon/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    meters = earth_radius * c
    return meters


