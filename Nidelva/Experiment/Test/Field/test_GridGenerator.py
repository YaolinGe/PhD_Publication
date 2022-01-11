from Nidelva.Experiment.Field.Grid.GridGenerator import GridGenerator
from Nidelva.Experiment.Field.Grid.GridConfig import GridConfig
from Nidelva.Experiment.Plotter.Scatter3dPlot import Scatter3DPlot
from gmplot import GoogleMapPlotter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from usr_func import *

pivot = 63.446905, 10.419426  # right bottom corner
angle_rotation = deg2rad(60)
nx = ny = 25
nz = 5
xlim = [0, 1000]
ylim = [0, 1000]
zlim = [0.5, 2.5]
gridConfig = GridConfig(pivot, angle_rotation, nx, ny, nz, xlim, ylim, zlim)

a = GridGenerator(gridConfig)
Scatter3DPlot(a.xyz_grid_usr, "grid_xyz")
Scatter3DPlot(a.grid_comparison, "grid_xyz_rotation_comparison")
Scatter3DPlot(a.xyz_grid_wgs, "grid_rotated")


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

plotGridonMap(a.WGScoordinages_grid)
import os
os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/MapPlot/map.html")




