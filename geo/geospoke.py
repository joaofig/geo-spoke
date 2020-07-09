import numpy as np
import pandas as pd
import h3.api.numpy_int as h3
import multiprocessing as mp

from geo.geomath import vec_haversine, num_haversine
from functools import partial, reduce


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
        dist = vec_haversine(self.lats, self.lons, lat, lon)
        return np.argwhere(dist <= r)

    def query_knn(self, location: np.array, k: int) -> np.array:
        dist = vec_haversine(self.lats, self.lons,
                             location[0, 0], location[0, 1])
        idx = np.argsort(dist)
        return idx[:k], dist[idx[:k]]


def spoke_init(param):
    dist = vec_haversine(param["lats"], param["lons"], param["lat"], param["lon"])
    idx = np.argsort(dist)
    sorted = dist[idx]
    return sorted, idx


# @jit(nopython=True)
def get_slice(dim: int, i: int, k: int) -> np.ndarray:
    return slice(max(0, i-k), min(dim-1, i+k)+1)


class GeoSpoke(object):

    def __init__(self, locations: np.ndarray):
        self.lats = locations[:, 0]
        self.lons = locations[:, 1]

        min_lat, max_lat = self.lats.min(), self.lats.max()
        min_lon, max_lon = self.lons.min(), self.lons.max()

        if max_lat > 0:
            self.lat0 = self.lat1 = min_lat - 90
        else:
            self.lat0 = self.lat1 = max_lat + 90
        self.lon0 = (max_lon - min_lon) / 2 - 45
        self.lon1 = self.lon0 + 90

        dist0 = vec_haversine(self.lats, self.lons, self.lat0, self.lon0)
        dist1 = vec_haversine(self.lats, self.lons, self.lat1, self.lon1)
        self.idx0 = np.argsort(dist0)
        self.idx1 = np.argsort(dist1)
        self.sorted0 = dist0[self.idx0]
        self.sorted1 = dist1[self.idx1]

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
        lat = location[0, 0]
        lon = location[0, 1]
        d0 = num_haversine(lat, lon, self.lat0, self.lon0)
        d1 = num_haversine(lat, lon, self.lat1, self.lon1)

        i0 = np.searchsorted(self.sorted0, d0 - r)
        i1 = np.searchsorted(self.sorted0, d0 + r)
        match0 = self.idx0[i0:i1+1]

        i0 = np.searchsorted(self.sorted1, d1 - r)
        i1 = np.searchsorted(self.sorted1, d1 + r)
        match1 = self.idx1[i0:i1+1]

        intersect = np.intersect1d(match0, match1)
        dist = vec_haversine(self.lats[intersect],
                             self.lons[intersect],
                             lat, lon)
        return intersect[dist <= r]

    def query_knn(self, location: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
        lat = location[0, 0]
        lon = location[0, 1]
        d0 = num_haversine(lat, lon, self.lat0, self.lon0)
        d1 = num_haversine(lat, lon, self.lat1, self.lon1)
        i0 = np.searchsorted(self.sorted0, d0)
        i1 = np.searchsorted(self.sorted1, d1)

        dim = self.idx0.shape[0]
        n = k
        intersect = np.zeros(0)
        while intersect.shape[0] < k:
            r0 = get_slice(dim, i0, n)
            r1 = get_slice(dim, i1, n)
            intersect = np.intersect1d(self.idx0[r0],
                                       self.idx1[r1],
                                       assume_unique=True)
            n *= 2

        dist0 = self.sorted0[r0]
        dist1 = self.sorted1[r1]
        r = max(d0 - dist0[0], dist0[-1] - d0,
                d1 - dist1[0], dist1[-1] - d1) * 1.5  # math.sqrt(2.0)

        i0 = np.searchsorted(self.sorted0, d0 - r)
        i1 = np.searchsorted(self.sorted0, d0 + r)
        match0 = self.idx0[i0:i1+1]

        i0 = np.searchsorted(self.sorted1, d1 - r)
        i1 = np.searchsorted(self.sorted1, d1 + r)
        match1 = self.idx1[i0:i1+1]

        intersect = np.intersect1d(match0, match1, assume_unique=True)
        dist = vec_haversine(self.lats[intersect],
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
        self.h3set = set(flattened)
        self.h3arr = np.array(flattened, dtype=np.int64)
        self.h3idx = np.argsort(self.h3arr)

    def query_radius(self,
                     location: np.ndarray,
                     r: float) -> np.ndarray:
        edge_len = h3.edge_length(self.h3res, unit="m")
        idx = h3.geo_to_h3(location[0, 0], location[0, 1], self.h3res)

        ring = self.h3set.intersection(h3.k_ring(idx, int(round(r / edge_len))))

        bool_ix = np.zeros_like(self.h3arr, dtype=np.bool)
        for hh in ring:
            bool_ix = bool_ix | (self.h3arr == hh)

        indices = np.argwhere(bool_ix).ravel()

        dist = vec_haversine(self.locations[indices, 0], self.locations[indices, 1],
                             location[0, 0], location[0, 1])
        return indices[np.argwhere(dist <= r).ravel()]

    def query_knn(self, location: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
        idx = h3.geo_to_h3(location[0, 0], location[0, 1], self.h3res)

        indices = []
        i = 0
        ring = set()
        while len(indices) < k:
            i += 1
            k_ring = set(h3.k_ring(idx, i))
            ring = k_ring - ring
            for hh in ring & self.h3set:
                indices.extend(np.argwhere(self.h3arr == hh).ravel().tolist())

        dist = vec_haversine(self.locations[indices, 0], self.locations[indices, 1],
                             location[0, 0], location[0, 1])
        idx = np.argsort(dist)
        return np.array(indices)[idx[:k]], dist[idx[:k]]


def main():
    import folium
    from folium.vector_layers import CircleMarker
    from timeit import default_timer as timer

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

    start = timer()
    geo_query = GeoSpoke(positions)
    end = timer()
    print("GeoSpoke initialization took {} seconds".format(end-start))

    start = timer()
    h3_index = H3Index(positions, resolution=11)
    end = timer()
    print("H3Index initialization took {} seconds".format(end-start))

    geo_brute = GeoBrute(positions)

    pt = np.array([[53.3520802, -6.2883607]])

    start = timer()
    ind = geo_query.query_radius(pt, r=100.0)
    end = timer()
    print(ind)
    print("GeoSpoke radius query took {} seconds".format(end - start))
    print("--------------")

    start = timer()
    ind = h3_index.query_radius(pt, r=100.0)
    end = timer()
    print(ind)
    print("H3Index radius query took {} seconds".format(end - start))
    print("--------------")

    print("KNN - GeoSpoke ------")
    start = timer()
    knn0, dist0 = geo_query.query_knn(pt, 5)
    end = timer()
    print("Timer: {}".format(end-start))
    print(knn0)
    print(dist0)
    print("--------------")

    print("KNN - H3Index ------")
    start = timer()
    knn0, dist0 = h3_index.query_knn(pt, 5)
    end = timer()
    print("Timer: {}".format(end-start))
    print(knn0)
    print(dist0)
    print("--------------")

    print("KNN - GeoBrute ------")
    start = timer()
    knn1, dist1 = geo_brute.query_knn(pt, 5)
    end = timer()
    print("Timer: {}".format(end-start))
    print(knn1)
    print(dist1)

    # m = folium.Map(location=pt)
    # for idx in knn0:
    #     CircleMarker(positions[idx], radius=1, color="#ff0000").add_to(m)
    # for idx in knn1:
    #     CircleMarker(positions[idx], radius=1, color="#0000ff").add_to(m)
    # CircleMarker(pt, radius=1, color="#000000").add_to(m)
    # m.save("./map.html")


if __name__ == "__main__":
    main()
