import numpy as np
import math


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
    earth_radius = 6371000

    rad_lat1 = np.radians(lat1)
    rad_lon1 = np.radians(lon1)
    rad_lat2 = np.radians(lat2)
    rad_lon2 = np.radians(lon2)

    dlon = rad_lon2 - rad_lon1
    dlat = rad_lat2 - rad_lat1

    a = np.sin(dlat/2.0)**2 + np.cos(rad_lat1) * np.cos(rad_lat2) \
        * np.sin(dlon/2.0)**2

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
    earth_radius = 6371000

    rad_lat1 = math.radians(lat1)
    rad_lon1 = math.radians(lon1)
    rad_lat2 = math.radians(lat2)
    rad_lon2 = math.radians(lon2)

    dlon = rad_lon2 - rad_lon1
    dlat = rad_lat2 - rad_lat1

    a = math.sin(dlat/2.0)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) \
        * np.sin(dlon/2.0)**2

    c = 2 * math.atan2(np.sqrt(a), math.sqrt(1.0 - a))
    meters = earth_radius * c
    return meters


