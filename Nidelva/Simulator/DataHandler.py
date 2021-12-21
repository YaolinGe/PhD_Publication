#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
from datetime import datetime
from usr_func import *


class RawDataHandler:

    datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Data/'
    # datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Adaptive/Data/'
    # figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Adaptive/fig/'

    def __init__(self):
        print("hello world")
        pass

    def load_raw_data(self):
        # % Data extraction from the raw data
        self.rawTemp = pd.read_csv(self.datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
        self.rawLoc = pd.read_csv(self.datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
        self.rawSal = pd.read_csv(self.datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
        self.rawDepth = pd.read_csv(self.datapath + "Depth.csv", delimiter=', ', header=0, engine='python')
        # rawGPS = pd.read_csv(datapath + "GpsFix.csv", delimiter=', ', header=0, engine='python')
        # rawCurrent = pd.read_csv(datapath + "EstimatedStreamVelocity.csv", delimiter=', ', header=0, engine='python')
        print("Raw data is loaded successfully!")

    def group_raw_data_based_on_timestamp(self):

        # To group all the time stamp together, since only second accuracy matters
        self.rawSal.iloc[:, 0] = np.ceil(self.rawSal.iloc[:, 0])
        self.rawTemp.iloc[:, 0] = np.ceil(self.rawTemp.iloc[:, 0])
        self.rawCTDTemp = self.rawTemp[self.rawTemp.iloc[:, 2] == 'SmartX'] # 'SmartX' because CTD sensor is SmartX
        self.rawLoc.iloc[:, 0] = np.ceil(self.rawLoc.iloc[:, 0])
        self.rawDepth.iloc[:, 0] = np.ceil(self.rawDepth.iloc[:, 0])
        self.rawDepth.iloc[:, 0] = np.ceil(self.rawDepth.iloc[:, 0])
        print("Raw data is grouped successfully!")

    def extract_all_data(self):
        self.depth_ctd = self.rawDepth[self.rawDepth.iloc[:, 2] == 'SmartX']["value (m)"].groupby(self.rawDepth["timestamp"]).mean()
        self.depth_dvl = self.rawDepth[self.rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(self.rawDepth["timestamp"]).mean()
        self.depth_est = self.rawLoc["depth (m)"].groupby(self.rawLoc["timestamp"]).mean()

        # indices used to extract data
        self.lat_origin = self.rawLoc["lat (rad)"].groupby(self.rawLoc["timestamp"]).mean()
        self.lon_origin = self.rawLoc["lon (rad)"].groupby(self.rawLoc["timestamp"]).mean()
        self.x_loc = self.rawLoc["x (m)"].groupby(self.rawLoc["timestamp"]).mean()
        self.y_loc = self.rawLoc["y (m)"].groupby(self.rawLoc["timestamp"]).mean()
        self.z_loc = self.rawLoc["z (m)"].groupby(self.rawLoc["timestamp"]).mean()
        self.depth = self.rawLoc["depth (m)"].groupby(self.rawLoc["timestamp"]).mean()
        self.time_loc = self.rawLoc["timestamp"].groupby(self.rawLoc["timestamp"]).mean()
        self.time_sal= self.rawSal["timestamp"].groupby(self.rawSal["timestamp"]).mean()
        self.time_temp = self.rawCTDTemp["timestamp"].groupby(self.rawCTDTemp["timestamp"]).mean()
        self.dataSal = self.rawSal["value (psu)"].groupby(self.rawSal["timestamp"]).mean()
        self.dataTemp = self.rawCTDTemp.iloc[:, -1].groupby(self.rawCTDTemp["timestamp"]).mean()
        print("Data is extracted successfully!")

    def save_data(self):
        #% Rearrange data according to their timestamp
        self.load_raw_data()
        self.group_raw_data_based_on_timestamp()
        self.extract_all_data()
        data = []
        time_mission = []
        x = []
        y = []
        z = []
        d = []
        sal = []
        temp = []
        lat = []
        lon = []

        for i in range(len(self.time_loc)):
            if np.any(self.time_sal.isin([self.time_loc.iloc[i]])) and np.any(self.time_temp.isin([self.time_loc.iloc[i]])):
                time_mission.append(self.time_loc.iloc[i])
                x.append(self.x_loc.iloc[i])
                y.append(self.y_loc.iloc[i])
                z.append(self.z_loc.iloc[i])
                d.append(self.depth.iloc[i])
                lat_temp = rad2deg(self.lat_origin.iloc[i]) + rad2deg(self.x_loc.iloc[i] * np.pi * 2.0 / circumference)
                lat.append(lat_temp)
                lon.append(rad2deg(self.lon_origin.iloc[i]) + rad2deg(self.y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_temp)))))
                sal.append(self.dataSal[self.time_sal.isin([self.time_loc.iloc[i]])].iloc[0])
                temp.append(self.dataTemp[self.time_temp.isin([self.time_loc.iloc[i]])].iloc[0])
            else:
                print(datetime.fromtimestamp(self.time_loc.iloc[i]))
                continue

        df = pd.DataFrame({"timestamp": time_mission, "lat": lat, "lon": lon, "x": x, "y": y, "z": z, "d": d, "sal": sal, "temp": temp})
        df.to_csv(self.datapath + "DataMerged.csv", index = False)
        print("Data is saved successfully!")

if __name__ == "__main__":
    a = RawDataHandler()
    a.save_data()

