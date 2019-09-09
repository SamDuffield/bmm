################################################################################
# Module: preprocess.py
# Description: Trim raw data by timestamp (if required - working on a small data
#              set initially is advised).
#              Trim and split to ensure all coordinates within a bounding box.
#              Convert longitude-latitude coordinates to UTM.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import data.utils
import os.path
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from pytz import timezone
import datetime


# Assume 15 seconds between coordinates (and trajectories are complete).
data_time_step = 15

# Define timezone (for converting times to/from UNIX timestamps)
# Here for Portugal
time_zone = timezone('Europe/Lisbon')

# Bounding box in longitude and latitude, [north (lat), south (lat), east (lon), west (lon)]
# Here for Porto, Portugal
bbox_ll = [41.176, 41.111, -8.563, -8.657]


def timestamp_end_extract(row):
    """
    Calculate the time the taxi completes it's trip using start timestamp and polyline/number of GPS pings.
    Requires predefined global variable data_time_step. Assumes trajectories are complete (no holes).
    :param row: row of data frame
    :return: timestamp taxi completes trip
    """
    polyline = row['POLYLINE']
    timestamp_poly_start = row['TIMESTAMP']
    timestamp_poly_end = timestamp_poly_start + data_time_step*max((len(polyline)-1), 0)
    return timestamp_poly_end


def datetime_to_timestamp(date_time: datetime.datetime):
    """
    Convert a datetime object (in local timezone) to a UNIX timestamp (seconds since 01 Jan 1970. (UTC))
    Requires predefined global timezone object (from pytz), time_zone.
    :param date_time: datetime object
    :return: UNIX timestamp (float)
    """
    # Ensure in correct timezone (important for converting to UNIX timestamp)
    local_date_time = time_zone.localize(date_time)

    # Convert to UNIX timestamp
    timestamp = local_date_time.timestamp()

    return timestamp


def timestamp_to_datetime(timestamp):
    """
    Convert a UNIX timestamp (seconds since 01 Jan 1970. (UTC)) to a datetime object (in local timezone).
    Requires predefined global timezone object (from pytz), time_zone.
    :param UNIX timestamp (float or int)
    :return: local_date_time: datetime object (local timezone)
    """
    # Get UTC datetime
    utc_dt = datetime.datetime.utcfromtimestamp(timestamp)

    # Make datetime object aware it is UTC
    utc_dt = utc_dt.replace(tzinfo=timezone('UTC'))

    # Change to local timezone
    local_date_time = utc_dt.astimezone(time_zone)

    return local_date_time


def coord_in_bbox(coordinate, bbox):
    """
    Checks whether a given coordinate is inside a given bounding box
    :param coordinate: [lat, long] or UTM as [y, x]
    :param bbox: [N, S, E, W] conformal to coordinate
    :return: boolean
    """
    return False if coordinate is None \
        else bbox[0] >= coordinate[1] >= bbox[1] and bbox[2] >= coordinate[0] >= bbox[3]


def index_inner_bbox(polyline, bbox):
    """
    Boolean list for coordinates inside bbox
    :param polyline: raw polyline
    :param bbox: bounding box conformal to polyline
    :return: booleans of coords inside bbox
    """
    return [coord_in_bbox(coordinate, bbox) for coordinate in polyline]


def poly_split_indices_bbox(polyline, bbox):
    """
    Indexes polyline and outputs segments to keep (discarding coords outside bbox or None).
    Will normally return one segment unless the taxi left bbox and returned in the same trip.
    :param polyline: raw polyline
    :param bbox: bounding box conformal to polyline
    :return: List of [start_index, end_index] for each retained segment
    """
    filtered_poly = index_inner_bbox(polyline, bbox)

    start = 0
    keep_indices = []
    for i, inside in enumerate(filtered_poly):
        if inside is False and i > 1 and filtered_poly[i-1] is True:
            keep_indices += [[start, i]]
        if inside is False:
            start = i + 1
        if i == len(filtered_poly) - 1 and inside is True:
            keep_indices += [[start, i + 1]]

    return keep_indices


def clean_row_segment(in_row, segment):
    """
    Given data row and a polyline segment, returns cleaned data row.
    New columns with potentially adjusted timestamp and cleaned raw and mm polylines
    :param in_row: input row of data frame
    :param segment:
    :return:
    """
    row = in_row.copy()
    row['TIMESTAMP'] = row['TIMESTAMP'] + segment[0]*data_time_step
    row['TIMESTAMP_END'] = row['TIMESTAMP'] + (segment[1] - segment[0] - 1) * data_time_step
    row['POLYLINE'] = row['POLYLINE'][segment[0]:segment[1]]

    return row


def clean_row_process(row, bbox):
    """
    Given data row, identifies splitting of polyline due to leaving bbox
    and returns subsequent split data rows.
    :param row: input row of data frame
    :param bbox: bounding box
    :return: data frame with potentially multiple rows due to polyline splitting
    """
    keep_indices = poly_split_indices_bbox(row['POLYLINE'], bbox)

    if len(keep_indices) == 0:
        return None

    out_df = pd.DataFrame()

    for segment in keep_indices:
        if (segment[1] - segment[0]) > 1:
            out_df = out_df.append([clean_row_segment(row, segment)])

    return out_df


def longlat_polys_to_utm(polylines, to_crs=None):
    """
    Takes a list of long-lat polylines and returns a list of UTM polylines
    :param polylines: pd.Series of [long, lat] elements
    :param to_crs: if crs already known (i.e. from projected OSMNx graph), not essential
    :return: pd.Series of [x,y] elements where x and y are metres from fixed UTM location
    """
    # Elongate polylines df, so that each row is a coordinate rather than a polyline
    # Additional columns for trip index and coordinate index
    trip_id = []
    coordinate_id = []
    longitude = []
    latitude = []

    trip_id_counter = 0
    for polyline in polylines:
        coordinate_id_counter = 0
        for coord in polyline:
            trip_id.append(trip_id_counter)
            coordinate_id.append(coordinate_id_counter)
            longitude.append(coord[0])
            latitude.append(coord[1])
            coordinate_id_counter += 1
        trip_id_counter += 1

    polylines_elong = gpd.GeoDataFrame({'trip_index': trip_id, 'coord_index': coordinate_id,
                                        'x': longitude, 'y': latitude})

    # Initiate with longlat crs
    polylines_elong.crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    # Add trivial geometry for conversion
    polylines_elong['geometry'] = polylines_elong.apply(lambda row: Point(row['x'], row['y']), axis=1)

    # Use OSMNx to convert to UTM
    polylines_elong_utm = ox.project_gdf(polylines_elong, to_crs=to_crs)

    # Extract UTM x and y values
    polylines_elong_utm['x'] = polylines_elong_utm['geometry'].map(lambda point: point.x)
    polylines_elong_utm['y'] = polylines_elong_utm['geometry'].map(lambda point: point.y)
    polylines_elong_utm = polylines_elong_utm.drop('geometry', axis=1)

    # Delongate data frame
    # Back to one row for each polyline
    polylines_utm = []
    for i in set(trip_id):
        trip_data = polylines_elong_utm[['x', 'y']][polylines_elong_utm['trip_index'] == i]
        polylines_utm += [trip_data.values.tolist()]
    polylines_utm = pd.Series(polylines_utm)

    return polylines_utm


if __name__ == '__main__':

    # Source data paths
    raw_path, process_data_path = data.utils.source_data()

    # Read data in chunks to save memory
    chunksize = 10 ** 3

    # Load raw data (csv reader)
    taxi_data = data.utils.read_data(raw_path, chunksize)

    # Number of chunks in raw data
    num_chunks = sum(1 for chunk in taxi_data)

    # Reinitialise generator
    taxi_data = data.utils.read_data(raw_path, chunksize)

    # Time interval to trim to
    time_start = datetime.datetime(2014, 5, 5, 0, 0, 0)
    time_end = datetime.datetime(2014, 5, 12, 0, 0, 0)

    # Convert to UNIX timestamp
    timestamp_start = datetime_to_timestamp(time_start)
    timestamp_end = datetime_to_timestamp(time_end)

    # Destination path for preprocessed data file
    save_data_path = process_data_path + '/data/'\
        + data.utils.project_title + '_'\
        + time_start.strftime("%d%m%Y") + '_'\
        + time_end.strftime("%d%m%Y") + '_'\
        + 'utm' + '_'\
        + 'bbox'\
        + '.csv'

    # Create data folder in process_data_path if it doesn't already exist
    if not os.path.exists(process_data_path + '/data/'):
        os.mkdir(process_data_path + '/data/')

    # Check if file already exists (as save_chunk appends)
    if os.path.isfile(save_data_path):
        print("File already exists at " + save_data_path)
        exit()

    # Progress counter
    chunk_count = 0

    # Iterate through data by chunks
    for chunk in taxi_data:
        # Only rows without missing data and more than location
        if 'MISSING_DATA' in chunk.columns:
            chunk = chunk[(~chunk['MISSING_DATA']) & (chunk['POLYLINE'].map(len) > 1)]
        else:
            chunk = chunk[chunk['POLYLINE'].map(len) > 1]

        # Add TIMESTAMP_END column
        chunk['TIMESTAMP_END'] = chunk.apply(timestamp_end_extract, axis=1)

        # Select rows with any coordinate in time interval
        chunk = chunk[(timestamp_start <= chunk['TIMESTAMP_END']) & (chunk['TIMESTAMP'] <= timestamp_end)]

        # Trim and split (by lon-lat) to remain in bbox
        # Initiate clean chunk
        chunk_clean = pd.DataFrame()

        # Clean rows (remove coords outside bbox, split if trajectory leaves and returns)
        for index, row in chunk.iterrows():
            clean_row = clean_row_process(row, bbox_ll)
            if clean_row is not None:
                chunk_clean = chunk_clean.append([clean_row], ignore_index=True)

        chunk = chunk_clean.copy()
        del chunk_clean

        if chunk.shape[0] > 0:
            # Convert polylines to utm
            chunk['POLYLINE_UTM'] = longlat_polys_to_utm(chunk['POLYLINE'])

            # Save
            data.utils.save_chunk(save_data_path, chunk)

        # Display progress
        chunk_count += 1
        print('\nCompleted ' + str(round(chunk_count/num_chunks, 3)*100) + '%\n')
