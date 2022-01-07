"""
This script integrates all the data together from different sensors
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-07
"""


import pandas as pd
from datetime import datetime
from usr_func import *
import warnings


class AUVDataIntegrator:

    def __init__(self, datapath, filename):
        if filename is None:
            raise ValueError(filename + " is not a valid filename, please check")
        self.datapath = datapath
        self.filename = filename
        self.save_data()

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

        time_mission = xauv = yauv = zauv = dauv = sal_auv = temp_auv = lat_auv = lon_auv = []

        for i in range(len(self.time_loc)):
            if np.any(self.time_sal.isin([self.time_loc.iloc[i]])) and np.any(self.time_temp.isin([self.time_loc.iloc[i]])):
                time_mission.append(self.time_loc.iloc[i])
                xauv.append(self.x_loc.iloc[i])
                yauv.append(self.y_loc.iloc[i])
                zauv.append(self.z_loc.iloc[i])
                dauv.append(self.depth.iloc[i])
                lat_temp = rad2deg(self.lat_origin.iloc[i]) + rad2deg(self.x_loc.iloc[i] * np.pi * 2.0 / circumference)
                lat_auv.append(lat_temp)
                lon_auv.append(rad2deg(self.lon_origin.iloc[i]) + rad2deg(self.y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_temp)))))
                sal_auv.append(self.dataSal[self.time_sal.isin([self.time_loc.iloc[i]])].iloc[0])
                temp_auv.append(self.dataTemp[self.time_temp.isin([self.time_loc.iloc[i]])].iloc[0])
            else:
                print(datetime.fromtimestamp(self.time_loc.iloc[i]))
                continue

        # ====== This section converts AUV coordinates from WGS84 back to rotated grid coordinates
        lat4, lon4 = 63.446905, 10.419426  # right bottom corner
        alpha = deg2rad(60)
        warnings.warn("The origin is set to be " + str(lat4) + str(lon4) + ", rotational angle is " + str(alpha) + "\n It only applies to case Nidelva")

        lat_auv = np.array(lat_auv).reshape(-1, 1)
        lon_auv = np.array(lon_auv).reshape(-1, 1)
        Dx = deg2rad(lat_auv - lat4) / 2 / np.pi * circumference
        Dy = deg2rad(lon_auv - lon4) / 2 / np.pi * circumference * np.cos(deg2rad(lat_auv))

        xauv = np.array(xauv).reshape(-1, 1)
        yauv = np.array(yauv).reshape(-1, 1)

        Rc = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        TT = (Rc @ np.hstack((Dx, Dy)).T).T
        xauv_new = TT[:, 0].reshape(-1, 1)
        yauv_new = TT[:, 1].reshape(-1, 1)

        zauv = np.array(zauv).reshape(-1, 1)
        dauv = np.array(dauv).reshape(-1, 1)
        sal_auv = np.array(sal_auv).reshape(-1, 1)
        temp_auv = np.array(temp_auv).reshape(-1, 1)
        time_mission = np.array(time_mission).reshape(-1, 1)

        lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv, time_mission = map(vectorise,
                                                                                        [lat_auv, lon_auv, xauv, yauv, zauv,
                                                                                         dauv, sal_auv, temp_auv, time_mission])

        self.datasheet = np.hstack((time_mission, lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv))
        # self.datasheet = np.hstack((time_mission, lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv))
        df = pd.DataFrame(self.datasheet, columns=["timestamp", "lat", "lon", "x", "y", "z", "depth", "salinity", "temperature"])
        df.to_csv(self.datapath+self.filename+".csv", index=False)
        # np.savetxt(figpath + "../data.txt", datasheet, delimiter = ",")
        print("Data is saved successfully!")
