

from Nidelva.Experiment.Data.AUVDataIntegrator import AUVDataIntegrator
import pandas as pd

auv_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Data/Data_on_July06.csv"

# TODO Make it modular and exchangable

# datasheet = AUVDataIntegrator(auv_datapath, "Data_on_July06")
datasheet = pd.read_csv(auv_datapath)

timestamp_mission = datasheet['timestamp']
lat_auv = datasheet['lat']
lon_auv = datasheet['lon']
x_auv = datasheet['x']
y_auv = datasheet['y']
z_auv = datasheet['z']
depth_auv = datasheet['depth']
sal_auv = datasheet['salinity']
temp_auv = datasheet['temperature']

import matplotlib.pyplot as plt
plt.plot(z_auv)
plt.show()



