
import os
import json

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

import bmm

from simulated_data.utils import rmse
from simulated_data.single_route_run import save_dir as load_dir


fl_pf_routes = np.load(load_dir + 'fl_pf.npy', allow_pickle=True)
fl_bsi_routes = np.load(load_dir + 'fl_bsi.npy', allow_pickle=True)
ffbsi_routes = np.load(load_dir + 'ffbsi.npy', allow_pickle=True)





