
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from usr_func import *

class MetHandler:

    wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/Met/wind_data.csv"
    tide_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/Met/tide_data.txt"

    def __init__(self):
        print("hello world")

    def load_wind(self):
        self.wind_data = pd.read_csv(self.wind_path, sep=";")
        self.wind_data = self.wind_data.iloc[:-1, :] # remove last row which does not have values
        timestamp = list(map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M").timestamp(), self.wind_data.iloc[:, 2]))
        self.wind_data["timestamp"] = timestamp
        wind_speed = list(map(lambda x: float(x.replace(",", ".")), self.wind_data["Middelvind"]))
        self.wind_data["wind_speed"] = wind_speed
        self.wind_data["wind_angle"] = self.wind_data["Vindretning"]
        print("Wind is loaded successfully!")

    def load_tide(self):
        self.tide_data = pd.read_csv(self.tide_path, sep="   ", skiprows=14, engine='python', header=None)
        self.tide_data.columns = ["timestring", "tide_measured(mm)", "tide_predicted(mm)"]
        timestamp = list(map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S+02:00").timestamp(), self.tide_data.iloc[:, 0]))
        self.tide_data["timestamp"] = timestamp
        self.tide_data = self.tide_data[["timestamp", "tide_measured(mm)", "tide_predicted(mm)", "timestring"]]
        print("Tide is loaded successfully!")

    def show_wind(self):
        u, v = self.ws2uv(self.wind_data["wind_speed"], self.wind_data["wind_angle"])
        plt.figure()
        plt.plot(u)
        plt.plot(v)
        plt.show()
        # plt.quiver(0, 0, )

    def angle2angle(self, angle):
        return 270 - angle

    def ws2uv(self, wind_speed, wind_angle): # convert wind speed / angle to u, v
        u = wind_speed * np.cos(deg2rad(self.angle2angle(wind_angle)))
        v = wind_speed * np.sin(deg2rad(self.angle2angle(wind_angle)))
        return u, v

    def uv2ws(self, u, v): # convert u, v to wind speed / angle
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        wind_angle = rad2deg(np.arctan2(v, u))
        return wind_speed, wind_angle

if __name__ == "__main__":
    a = MetHandler()
    # a.load_wind()
    # a.load_tide()
    # a.show_wind()

#%%
# path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/Met/wind_data.csv"
# data = pd.read_csv(path, sep=";")
# data = data.iloc[:-1, :]
#
# timestamp = list(map(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M").timestamp(), data.iloc[:, 2]))
#
# print(np.array(data["Middelvind"]))
# wind_speed = list(map(lambda x: float(x.replace(",", ".")), data["Middelvind"]))
# print(np.array(data["Vindretning"]))


