import numpy as np
import pandas as pd
import h3.api.numpy_int as h3
import multiprocessing as mp
import math
import geo.geomath as gm

from functools import partial
from timeit import default_timer as timer


class GeoBrute(object):

    def __init__(self, locations: np.ndarray):
        self.lats = locations[:, 0]
        self.lons = locations[:, 1]

    def query_radius(self,
                     location: np.ndarray,
                     r: float) -> (np.ndarray, np.ndarray):
        """
        Selects the indices of the points that lie within a given distance from
        a given location.
        :param location: Location to query in [lat, lon] format
        :param r: Radius in meters
        :return: Array of indices
        """
        lat = location[0, 0]
        lon = location[0, 1]
        dist = gm.vec_haversine(self.lats, self.lons, lat, lon)
        return np.argwhere(dist <= r)

    def query_knn(self, location: np.array, k: int) -> np.array:
        dist = gm.vec_haversine(self.lats, self.lons,
                                location[0], location[1])
        idx = np.argsort(dist)
        return idx[:k], dist[idx[:k]]


def get_slice(dim: int, i: int, k: int) -> np.ndarray:
    return slice(max(0, i - k), min(dim - 1, i + k) + 1)


def calculate_sorted_distances(latitudes, longitudes, lat, lon):
    dist = gm.vec_haversine(latitudes, longitudes, lat, lon)
    idx = np.argsort(dist)
    return idx, dist[idx]


class GeoSpoke(object):

    def __init__(self, locations: np.ndarray):
        self.lats = locations[:, 0]
        self.lons = locations[:, 1]

        min_lat, max_lat = self.lats.min(), self.lats.max()
        min_lon, max_lon = self.lons.min(), self.lons.max()

        h = gm.num_haversine(min_lat, min_lon, max_lat, min_lon)
        w = gm.num_haversine(min_lat, min_lon, min_lat, max_lon)

        self.density = locations.shape[0] / (w * h)

        if max_lat > 0:
            self.lat0 = self.lat1 = min_lat - 90
        else:
            self.lat0 = self.lat1 = max_lat + 90
        self.lon0 = (max_lon - min_lon) / 2 - 45
        self.lon1 = self.lon0 + 90

        self.idx0, self.sorted0 = calculate_sorted_distances(self.lats, self.lons, self.lat0, self.lon0)
        self.idx1, self.sorted1 = calculate_sorted_distances(self.lats, self.lons, self.lat1, self.lon1)

    def query_radius(self,
                     location: np.ndarray,
                     r: float) -> np.ndarray:
        """
        Selects the indices of the points that lie within a given distance from
        a given location.
        :param location: Location to query in [lat, lon] format
        :param r: Radius in meters
        :return: Array of indices
        """
        lat = location[0]
        lon = location[1]
        d0 = gm.num_haversine(lat, lon, self.lat0, self.lon0)
        d1 = gm.num_haversine(lat, lon, self.lat1, self.lon1)

        i0 = np.searchsorted(self.sorted0, d0 - r)
        i1 = np.searchsorted(self.sorted0, d0 + r)
        match0 = self.idx0[i0:i1 + 1]

        i0 = np.searchsorted(self.sorted1, d1 - r)
        i1 = np.searchsorted(self.sorted1, d1 + r)
        match1 = self.idx1[i0:i1 + 1]

        intersect = np.intersect1d(match0, match1)
        dist = gm.vec_haversine(self.lats[intersect],
                                self.lons[intersect],
                                lat, lon)
        return intersect[dist <= r]

    def query_knn(self, location: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
        lat = location[0]
        lon = location[1]
        d0 = gm.num_haversine(lat, lon, self.lat0, self.lon0)
        d1 = gm.num_haversine(lat, lon, self.lat1, self.lon1)
        r = math.sqrt(k / self.density) * 2.0

        intersect = np.zeros(0)
        while intersect.shape[0] < k:
            s0 = np.searchsorted(self.sorted0, [d0 - r, d0 + r])
            s1 = np.searchsorted(self.sorted1, [d1 - r, d1 + r])
            intersect = np.intersect1d(self.idx0[s0[0]:s0[1] + 1],
                                       self.idx1[s1[0]:s1[1] + 1],
                                       assume_unique=True)
            r *= 4

        dist = gm.vec_haversine(self.lats[intersect],
                                self.lons[intersect],
                                lat, lon)

        idx = np.argsort(dist)
        return intersect[idx][:k], dist[idx[:k]]


def geo_to_h3_array(locations, resolution: int = 12):
    hexes = [h3.geo_to_h3(locations[i, 0], locations[i, 1], resolution) for i in range(locations.shape[0])]
    return hexes


class H3Index(object):

    def __init__(self, locations: np.ndarray, resolution=10):
        self.locations = locations
        self.h3res = resolution
        cpus = mp.cpu_count()
        arrays = np.array_split(locations, cpus)
        fn = partial(geo_to_h3_array, resolution=resolution)
        with mp.Pool(processes=cpus) as pool:
            results = pool.map(fn, arrays)
        flattened = [item for sublist in results for item in sublist]
        self.h3arr = np.array(flattened, dtype=np.uint64)
        self.h3idx = np.argsort(self.h3arr)

    def query_radius(self,
                     location: np.ndarray,
                     r: float) -> np.ndarray:
        edge_len = h3.edge_length(self.h3res, unit="m")
        idx = h3.geo_to_h3(location[0], location[1], self.h3res)

        ring = h3.k_ring(idx, 1 + int(round(r / edge_len)))

        i0 = np.searchsorted(self.h3arr, ring, side='left', sorter=self.h3idx)
        i1 = np.searchsorted(self.h3arr, ring, side='right', sorter=self.h3idx)

        indices = np.hstack([np.arange(i, j) for i, j in zip(i0, i1) if i != j])

        idx = self.h3idx[indices]
        dist = gm.vec_haversine(self.locations[idx, 0], self.locations[idx, 1],
                                location[0], location[1])
        return self.h3idx[indices[np.argwhere(dist <= r).ravel()]]

    def query_knn(self, location: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
        idx = h3.geo_to_h3(location[0], location[1], self.h3res)

        i = 0
        indices = np.zeros(0, dtype=np.uint64)
        ring = np.zeros(0, dtype=np.uint64)
        while indices.shape[0] < k:
            i += 2
            k_ring = h3.k_ring(idx, i)
            ring = np.setdiff1d(k_ring, ring, assume_unique=True)

            i0 = np.searchsorted(self.h3arr, ring, side='left', sorter=self.h3idx)
            i1 = np.searchsorted(self.h3arr, ring, side='right', sorter=self.h3idx)

            indices = np.hstack((indices,
                                 np.hstack([np.arange(i, j, dtype=np.uint64)
                                            for i, j in zip(i0, i1) if i != j])))

        idx = self.h3idx[indices]
        dist = gm.vec_haversine(self.locations[idx, 0],
                                self.locations[idx, 1],
                                location[0], location[1])

        dist_idx = np.argsort(dist)
        return idx[dist_idx[:k]], dist[dist_idx[:k]]


def main():
    import folium
    from folium.vector_layers import CircleMarker
    
    # np.random.randint(111)

    """
    For testing purposes only
    :return:
    """
    columns_to_read = ['Timestamp', 'LineID', 'Direction', 'PatternID',
                       'JourneyID', 'Congestion', 'Lon', 'Lat',
                       'Delay', 'BlockID', 'VehicleID', 'StopID', 'AtStop']
    df = pd.read_parquet("../data/sir010113-310113.parquet",
                         columns=columns_to_read)

    positions = df[['Lat', 'Lon']].to_numpy()

    random_indexes = np.random.randint(low=0, high=positions.shape[0], size=100)
    random_locations = positions[random_indexes]

    start = timer()
    geo_query = GeoSpoke(positions)
    end = timer()
    print("GeoSpoke initialization took {} seconds".format(end - start))

    start = timer()
    h3_index = H3Index(positions, resolution=10)
    end = timer()
    print("H3Index initialization took {} seconds".format(end - start))

    geo_brute = GeoBrute(positions)
    #
    # pt = np.array([[53.3520802, -6.2883607]])

    ind = np.zeros(0)

    start = timer()
    for pt in random_locations:  # [random_locations[0]]:
        ind = geo_query.query_radius(pt, r=100.0)
    end = timer()
    print(ind.shape[0], np.sort(ind))
    print("GeoSpoke radius query took {} seconds".format(end - start))
    print("--------------")

    start = timer()
    for pt in random_locations:  # [random_locations[0]]:
        ind = h3_index.query_radius(pt, r=100.0)
    end = timer()
    print(ind.shape[0], np.sort(ind))
    print("H3Index radius query took {} seconds".format(end - start))
    print("--------------")
    print(" ")

    print("KNN - GeoSpoke ------")
    start = timer()
    for pt in random_locations:
        knn0, dist0 = geo_query.query_knn(pt, 20)
    end = timer()
    print("Timer: {}".format(end - start))
    print(knn0)
    print(dist0)
    print("--------------")
    print(" ")

    print("KNN - H3Index ------")
    start = timer()
    for pt in random_locations:
        knn0, dist0 = h3_index.query_knn(pt, 20)
    end = timer()
    print("Timer: {}".format(end - start))
    print(knn0)
    print(dist0)
    print("--------------")

    # print("KNN - GeoBrute ------")
    # start = timer()
    # knn1, dist1 = geo_brute.query_knn(random_locations[0].ravel(), 20)
    # end = timer()
    # print("Timer: {}".format(end - start))
    # print(knn1)
    # print(dist1)

    # m = folium.Map(location=pt)
    # for idx in knn0:
    #     CircleMarker(positions[idx], radius=1, color="#ff0000").add_to(m)
    # for idx in knn1:
    #     CircleMarker(positions[idx], radius=1, color="#0000ff").add_to(m)
    # CircleMarker(pt, radius=1, color="#000000").add_to(m)
    # m.save("./map.html")


if __name__ == "__main__":
    main()
