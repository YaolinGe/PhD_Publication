import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

# df = pd.read_csv('../input/ukTrafficAADF.csv')
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(llcrnrlon=10.,llcrnrlat=63.40,
            urcrnrlon=10.45,urcrnrlat=63.45,
            resolution='i', # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing
            projection='tmerc', # The projection style is what gives us a 2D view of the world for this
            lon_0=10.41,lat_0=63.42, # Setting the central point of the image
            epsg=27700) # Setting the coordinate system we're using

m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like
# m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like
# m.drawcoastlines()
# m.drawrivers() # Default colour is black but it can be customised
m.drawcountries()

#%%

import netCDF4
import matplotlib
# Matplotlib plotting backend. Change to produce other
# filetypes. See documentation for alternatives.
# matplotlib.use('pdf')
# Basemap is a library that lets us to draw on a map.
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

# Setting up the map
# We use stereographic projection centered on 65 degrees
# north, 12 degrees east. (Approximately the location of
# Bindal kommune, on the Helgeland coast). Width and
# height is set to cover a bit more than just Norway
print('Creating map')
m = Basemap(width=1800000,height=2300000,
    resolution='h',projection='stere',lat_ts=65,lat_0=65,lon_0=12)

# Draw continents, from the database that comes with
# matplotlib as well as country borders and a border
# around the map
m.fillcontinents(color='#aaaaaa',lake_color='#cccccc')
m.drawcountries(linewidth=0.2)
m.drawmapboundary(fill_color='#cccccc')

# draw parallels and meridians for every 20 degrees.
# Most of these won't show up in the map selection used
# here, but that doesn't matter.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))

# met.no provides us with free data (or rather, free as
# in 200 free texts, since we pay for the service with
# our taxes), downloadable through the magic of thredds.
# You can open this url as if it was a local file, and
# the data are only downloaded when needed, and only as
# much as needed. Note that this link will remain valid
# only for a limited time. See the website for
# available files at any given time.
# dataurl = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2014080700.nc'
dataurl = "https://thredds.met.no/thredds/catalog/fou-hi/norkyst800m/catalog.html?dataset=norkyst800m_24h_files/NorKyst-800m_ZDEPTHS_avg.fc.2022011900.nc"
data = netCDF4.Dataset(dataurl)

print('Dowloading data')
# The full dataset is pretty large, so we will download
# only every third datapoint in either direction.
dx, dy = 3, 3
# The dimensions of the dataset is time, depth, lon, lat.
# We only download the first timestep, only the top layer
# and only every third datapoint in the horisontal
# directions.
temp = data.variables['temperature'][0,0,::dx,::dy]
# gridlons and gridlats tell us the lon and lat of each
# point in the temperature grid. This (or other,
# equivalent information) is necessary for drawing on
# the map. These are two dimensional grids and we only
# download every third element in either direction to
# match the temperature data.
gridlons = data.variables['lon'][::dx,::dy]
gridlats = data.variables['lat'][::dx,::dy]

# Here we use information from the map projection to
# convert from lon and lat to coordinates that can be
# used to draw data onto the map.
X, Y = m(gridlons, gridlats)

print('Plotting')
# pcolormesh is a function which plots an array of data
# as a grid of cells with color according to the value.
plt.pcolormesh(X, Y, temp)
# Add a colorbar, so we can read the temperature.
plt.colorbar()

print('Saving')
# Save the plot to a file.
# This can take a while, and produces a 12 MB pdf. To
# create a smaller file, such as a .png, change the
# filename as well as the backend.
# plt.savefig('map.pdf')
plt.show()


#%%
datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2020.05.01.nc"
import netCDF4
from usr_func import *
import matplotlib
matplotlib.use('macosx')
import numpy as np
data = netCDF4.Dataset(datapath, 'r')
lat = np.array(data["gridLats"])
lon = np.array(data["gridLons"])
salinity = np.array(np.nanmean(data["salinity"][:, 0, :, :], axis = 0))
import matplotlib.pyplot as plt
# plt.scatter(lon, lat, c=salinity, cmap="Paired", vmin=10, vmax=30)
# plt.colorbar()
# plt.show()

import pandas as pd
df = pd.DataFrame(np.hstack((vectorise(lat), vectorise(lon), vectorise(salinity))), columns=["lat", "lon", "salinity"])
df.to_csv("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2020.05.01_surface_mean_salinity.csv", index=False)



