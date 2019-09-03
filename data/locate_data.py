################################################################################
# Module: locate_data.py
# Description: Locate raw data and directory to save processed data.
#              Store paths in data/data_source,
#              with function to load again later.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import os
import json
import tkinter
from tkinter import filedialog


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
    :param raw: boolean whether to load raw data file path
    :param processed: boolean whether to load processed data directory path
    :return: path(s) as string(s)
    """
    source_file = open("data/data_source", "r")
    json_dict = source_file.read()
    source_file.close()

    data_dict = json.loads(json_dict)

    return data_dict.values()


if __name__ == '__main__':
    # User inputs raw file and process directory
    raw_path, process_path = locate_data_paths()

    # Write to data/data_source
    write_to_data_source(raw_path, process_path)

    # Load data with
    # raw_path, process_data_path = source_data()
    # or
    # _, process_data_path = source_data()
