import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/transfer_145711_files_ee6457cf/"

files = os.listdir(path_data)
for file in files:
    sinmod = netCDF4.Dataset(path_data + file)

class SINMOD:

    def __init__(self):
        print("Hello world")

    def load_data(self):



