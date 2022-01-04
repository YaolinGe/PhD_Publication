
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from gmplot import GoogleMapPlotter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.pyplot as plt

polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                   [6.344800000000000040e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.040000000000000036e+01]])
grid = GridGenerator(polygon = polygon, distance_neighbour = 120, no_children=6).grid

plt.plot(grid[:, 1], grid[:, 0], 'k.')
plt.show()

def plotGridonMap(grid):
    def color_scatter(gmap, lats, lngs, values=None, colormap='coolwarm',
                      size=None, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3])  # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            colors = [None for _ in lats]
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            gmap.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker, s=s, **kwargs)

    initial_zoom = 12
    apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
    gmap = GoogleMapPlotter(grid[0, 0], grid[0, 1], initial_zoom, apikey=apikey)
    color_scatter(gmap, grid[:, 0], grid[:, 1], np.zeros_like(grid[:, 0]), size=20, colormap='hsv')
    gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

plotGridonMap(grid)
import os
os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/MapPlot/map.html")


