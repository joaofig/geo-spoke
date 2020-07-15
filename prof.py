import numpy as np
import pandas as pd
import geo.geospoke as gs
import cProfile


def main():
    columns_to_read = ['Timestamp', 'LineID', 'Direction', 'PatternID',
                       'JourneyID', 'Congestion', 'Lon', 'Lat',
                       'Delay', 'BlockID', 'VehicleID', 'StopID', 'AtStop']
    df = pd.read_parquet("data/sir010113-310113.parquet",
                         columns=columns_to_read)

    positions = df[['Lat', 'Lon']].to_numpy()

    random_indexes = np.random.randint(low=0, high=positions.shape[0], size=100)
    random_locations = positions[random_indexes]

    h3_index = gs.H3Index(positions, resolution=10)

    ind = None
    for pt in random_locations:
        ind = h3_index.query_radius(pt, r=100.0)


if __name__ == "__main__":
    cProfile.run('main()')
