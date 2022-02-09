"""
This script plots the knowledge into slices
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-06
"""


from usr_func import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Plot2D:

    def __init__(self, knowledge=None, vmin=28, vmax=30, filename="mean"):
        if knowledge is None:
            raise ValueError("")
        self.knowledge = knowledge
        self.coordinates = self.knowledge.coordinates
        self.vmin = vmin
        self.vmax = vmax
        self.filename = filename
        self.plot()

    def plot(self):
        lat = self.coordinates[:, 0]
        lon = self.coordinates[:, 1]
        depth = self.coordinates[:, 2]
        depth_layer = np.unique(depth)
        number_of_plots = len(depth_layer)

        fig = plt.figure(figsize=(6*number_of_plots, 6))
        gs = GridSpec(nrows=1, ncols=number_of_plots)
        nx = 100
        ny = 100

        for i in range(number_of_plots):

            ind_layer = np.where(depth == depth_layer[i])[0]
            ax = fig.add_subplot(gs[i])
            grid_x, grid_y, grid_value = interpolate_2d(lon[ind_layer], lat[ind_layer], nx=nx, ny=ny,
                                                        value=self.knowledge.mu[ind_layer], interpolation_method="cubic")
            ax.grid(alpha=.2, linestyle='-.', color='k')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.set_xticks(np.arange(np.amin(lon), np.amax(lon), step=.005))
            ax.set_yticks(np.arange(np.amin(lat), np.amax(lat), step=.005))
            ax.set_ylim([63.4489, 63.4592])
            ax.set_xlim([10.4022, 10.418])

            if ax.is_first_col():
                ax.set_ylabel('Latitude')
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

            # if ax.is_first_row():
            #     ax.set_xlabel('Longitude')
            # else:
            #     plt.setp(ax.get_xticklabels(), visible=False)

            ax.set_title("Salinity at {:.1f} metre depth".format(depth_layer[i]))
            discrete_cmap = plt.get_cmap("BrBG", 10)
            # im = ax.scatter(grid_x, grid_y, c=grid_value, vmin=self.vmin, vmax=self.vmax, cmap="BrBG")
            im = ax.scatter(grid_x, grid_y, c=grid_value, vmin=0, vmax=30, cmap=discrete_cmap)

        axins = inset_axes(ax,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0, 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )

        cbar = plt.colorbar(im, cax=axins, pad=.05)
        cbar.ax.set_ylabel("Salinity (ppt)", rotation=-90, labelpad=30)
        plt.savefig("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/SINMOD2DSlices.pdf",
                    dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.show()



