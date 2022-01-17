
import numpy as np
import matplotlib.pyplot as plt
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from usr_func import *
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc


polygon = np.array([[0, 0],
                    [0, .001],
                    [.001, .001],
                    [.001, 0],
                    [0, 0]])

gridGenerator = GridGenerator(polygon, depth=[0], distance_neighbour=20)
coordinates = gridGenerator.coordinates
path = [3, 0, 1, 5, ]
lat_now, lon_now = coordinates[5, 0:2]
x_dist, y_dist = latlon2xy(coordinates[:, 0], coordinates[:, 1], lat_now, lon_now)
dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
ind_cand = np.argsort(dist)[:7]
ind_cand = ind_cand[ind_cand != 5]
ind_not_selected = [2, 4, 1]

fig = plt.figure(figsize=(15, 4), )
gs = GridSpec(nrows=1, ncols=5)

ax = fig.add_subplot(gs[0])
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(coordinates[path, 1], coordinates[path, 0], 'y')
plt.plot(coordinates[5, 1], coordinates[5, 0], 'r.', markersize=10)
plt.gca().axis("off")
plt.title("I")


ax = fig.add_subplot(gs[1])
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(coordinates[path, 1], coordinates[path, 0], 'y-')
plt.plot(coordinates[ind_cand, 1], coordinates[ind_cand, 0], 'c*')
plt.plot(coordinates[5, 1], coordinates[5, 0], 'r.', markersize=10)
arc = Arc((coordinates[5, 1], coordinates[5, 0]), .00039,.00039, 30)
arc.set_linestyle("--")
plt.gca().add_patch(arc)
plt.gca().axis("off")
plt.title("II")


ax = fig.add_subplot(gs[2])
ind_cand = ind_cand[ind_cand!=2]
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(coordinates[path[:-1], 1], coordinates[path[:-1], 0], 'y-')
plt.plot(coordinates[ind_cand, 1], coordinates[ind_cand, 0], 'c*')
plt.plot(coordinates[5, 1], coordinates[5, 0], 'r.', markersize=10)

U = coordinates[5, 1] - coordinates[1, 1] - .00001
V = coordinates[5, 0] - coordinates[1, 0] + .00001
plt.quiver(coordinates[1, 1], coordinates[1, 0], U, V, scale_units='xy', scale=1., color='b')

U = coordinates[ind_not_selected[0], 1] - coordinates[5, 1]
V = coordinates[ind_not_selected[0], 0] - coordinates[5, 0]
plt.quiver(coordinates[5, 1], coordinates[5, 0], U, V, scale_units='xy', scale=1., color='b')
arc = Arc((coordinates[5, 1], coordinates[5, 0]), .0001,.0001, theta1=180, theta2=240, color='b')
arc.set_linestyle("-")
plt.gca().add_patch(arc)

plt.plot(coordinates[2, 1], coordinates[2, 0], 'bx', markersize=10)
plt.gca().axis("off")
plt.title("III")


ax = fig.add_subplot(gs[3])
ind_cand = ind_cand[ind_cand!=4]
ind_cand = ind_cand[ind_cand!=1]
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(coordinates[path[:-1], 1], coordinates[path[:-1], 0], 'y-')
plt.plot(coordinates[ind_cand[ind_cand != ind_cand[1]], 1], coordinates[ind_cand[ind_cand != ind_cand[1]], 0], 'c*')
plt.plot(coordinates[ind_cand[1], 1], coordinates[ind_cand[1], 0], 'gX', markersize=10)
plt.plot(coordinates[5, 1], coordinates[5, 0], 'r.', markersize=10)
plt.plot(coordinates[ind_not_selected, 1], coordinates[ind_not_selected, 0], 'bx', markersize=10)

U = coordinates[5, 1] - coordinates[1, 1] - .00001
V = coordinates[5, 0] - coordinates[1, 0] + .00001
plt.quiver(coordinates[1, 1], coordinates[1, 0], U, V, scale_units='xy', scale=1., color='g')
U = coordinates[ind_cand[1], 1] - coordinates[5, 1]
V = coordinates[ind_cand[1], 0] - coordinates[5, 0]
plt.quiver(coordinates[5, 1], coordinates[5, 0], U, V, scale_units='xy', scale=1., color='g')
arc = Arc((coordinates[5, 1], coordinates[5, 0]), .0001,.0001, theta1=-120, theta2=0, color='g')
arc.set_linestyle("-")
plt.gca().add_patch(arc)
plt.gca().axis("off")
plt.title("IV")


ax = fig.add_subplot(gs[4])
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.plot(coordinates[path, 1], coordinates[path, 0], 'y')
plt.plot(coordinates[5, 1], coordinates[5, 0], 'r.', markersize=10)
plt.plot(coordinates[ind_cand, 1], coordinates[ind_cand, 0], 'gX', markersize=10)
plt.plot(coordinates[ind_not_selected, 1], coordinates[ind_not_selected, 0], 'bx', markersize=10)
plt.gca().axis("off")
plt.title("V")


plt.tight_layout()
plt.savefig("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/PathPlanning/smooth_filtering.pdf")
plt.show()




