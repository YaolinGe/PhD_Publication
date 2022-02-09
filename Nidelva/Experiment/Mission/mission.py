"""
This script contains mission description
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-09
"""
from usr_func import *


class mission:

    # datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/Data/Salinity.csv"
    # datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Salinity.csv"
    datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Adaptive/Salinity.csv"

    def __init__(self):
        self.get_mission_duration()
        pass

    def get_mission_duration(self):
        df = pd.read_csv(self.datapath)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        print(df)
        pass


a = mission()


