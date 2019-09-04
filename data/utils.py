################################################################################
# Module: utils.py
# Description: Locate raw data and directory to save processed data (so we can
#              work with the same raw data on different computers/disks).
#              Store selected paths in data/data_source, along with functions
#              to load paths, read data and save data.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import os
import json
import tkinter
from tkinter import filedialog
import pandas as pd


# Data naming convention (not for raw data, but for anything after)
# [project_title]_[startdate]_[enddate]_[ll or utm]_[anything else].csv
# Date format: "%d%m%Y" e.g 01012000
project_title = 'portotaxi'


def locate_data_paths():
    """
    Locate both raw data file and directory for storing processed data.
    :return: [raw path str, process dir str]
    """
    root = tkinter.Tk()
    root.wm_withdraw()
    raw_file = filedialog.askopenfilename(initialdir="/", title='Locate raw data')
    process_dir = filedialog.askdirectory(initialdir=os.path.dirname(raw_file),
                                          title='Choose folder to save processed data')
    root.destroy()
    return [raw_file, process_dir]


def write_to_data_source(raw_file, process_dir):
    """
    Write raw and process paths to source file (JSON'd dict)
    :param raw_file: raw file path
    :param process_dir: process dir path
    """
    json_dict = json.dumps({'raw_data_file': raw_file,
                            'process_data_dir': process_dir})

    source_file = open("data_source", "w")
    source_file.write(json_dict)
    source_file.close()


def source_data():
    """
    Load file path for raw data and/or directory path for processed data.
    :return: [raw path, process path] as strings
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    source_file = open(curdir + "/data_source", "r")
    json_dict = source_file.read()
    source_file.close()

    data_dict = json.loads(json_dict)

    return data_dict.values()


def read_data(path, chunksize=None):
    """
    Read vehicle trajectory data from csv, automatically converting polylines stored as JSON
    :param path: location of csv
    :param chunksize: size of chunks to iterate through, None loads full file at once
    :return: generator iterating over (chunks of) data from csv
    """
    data_reader = pd.read_csv(path, chunksize=10)
    data_columns = data_reader.get_chunk().columns
    polyline_converters = {col_name: json.loads for col_name in data_columns
                           if 'POLYLINE' in col_name}

    return pd.read_csv(path, converters=polyline_converters, chunksize=chunksize)


def save_chunk(path, chunk):
    """
    Save (chunk of) data to csv, appending if file already exists
    :param chunk: data to be saved
    :param path: destination of csv
    """
    for col_name in chunk.columns:
        if 'POLYLINE' in col_name:
            chunk[col_name] = chunk[col_name].apply(json.dumps)

    csv_exists = os.path.isfile(path)

    chunk.to_csv(path, mode='a', index=False, header=not csv_exists)


def choose_data():
    """
    Prompts file dialog to ask user to locate data file.
    :return: path of chosen file (string)
    """
    _, process_path = source_data()
    root = tkinter.Tk()
    root.wm_withdraw()
    data_path = filedialog.askopenfilename(initialdir=process_path + "/data/", title='Locate data')
    root.destroy()
    return data_path


if __name__ == '__main__':

    # User inputs raw file and process directory
    raw_path, process_path = locate_data_paths()

    # Write to data/data_source
    write_to_data_source(raw_path, process_path)

    # Load data paths with
    # import data.utils
    # raw_path, process_data_path = data.utils.source_data()
    # or
    # _, process_data_path = data.utils.source_data()

    # Locate a data file with
    # data_path = data.utils.choose_data()
