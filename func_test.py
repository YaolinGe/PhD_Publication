import numpy as np
import matplotlib.pyplot as plt

import arcgis
from arcgis.gis import GIS

gis = GIS()
map1 = gis.map("Norway")
map1
plt.show()




