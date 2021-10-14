#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not
import plotly

plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

'''
Goal of the script is to make the class structure
Grid: generate the grid
GP: take care of the processes
Path planner: plan the next waypoint

This grid generation will generate the polygon grid as desired, using non-binary tree with recursion, it is very efficient
'''
os.system("find . -empty -type d -delete") # check empty folders and remove them to save space


class WaypointNode:
    '''
    generate node for each waypoint
    '''
    waypoint_loc = None
    subwaypoint_len = 0
    subwaypoint_loc = []

    def __init__(self, subwaypoints_len, subwaypoints_loc, waypoint_loc):
        self.subwaypoint_len = subwaypoints_len
        self.subwaypoint_loc = subwaypoints_loc
        self.waypoint_loc = waypoint_loc


class GridPoly(WaypointNode):
    '''
    generate the polygon grid with equal-distance from one to another
    '''
    lat_origin, lon_origin = 0, 0  # the right bottom corner coordinates
    distance_poly = 1  # [m], distance between two neighbouring points
    pointsPr = 1000  # points per layer
    polygon = None
    loc_start = None
    counter_plot = 0  # counter for plot number
    counter_grid = 0  # counter for grid points
    debug = True
    voiceCtrl = False
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/grid/"

    def __init__(self, polygon=np.array([[0, 0],
                                         [0, 1000],
                                         [1000, 1000],
                                         [1000, 0],
                                         [0, 0]]), debug=True, voiceCtrl=False):
        if debug:
            self.checkFolder()
        self.grid_poly = []
        self.polygon = polygon
        self.debug = debug
        self.voiceCtrl = voiceCtrl
        self.polygon_path = mplPath.Path(self.polygon)
        self.angle_poly = self.deg2rad(np.arange(0, 6) * 60)  # angles for polygon
        self.getPolygonArea()

        print("Grid polygon is activated!")
        print("Distance between neighbouring points: ", self.distance_poly)
        print("Starting location: ", self.loc_start)
        print("Polygon: ", self.polygon.shape)
        print("Points desired: ", self.pointsPr)
        print("Debug mode: ", self.debug)
        print("fig path: ", self.figpath)
        t1 = time.time()
        self.getGridPoly()
        t2 = time.time()
        print("Grid discretisation takes: {:.2f} seconds".format(t2 - t1))

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def revisit(self, loc):
        '''
        func determines whether it revisits the points it already have
        '''
        temp = np.array(self.grid_poly)
        if len(self.grid_poly) > 0:
            dist_min = np.min(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            ind = np.argmin(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            if dist_min <= .00001:
                return [True, ind]
            else:
                return [False, []]
        else:
            return [False, []]

    def getNewLocations(self, loc):
        '''
        get new locations around the current location
        '''
        x_delta = self.distance_poly * np.sin(self.angle_poly)
        y_delta = self.distance_poly * np.cos(self.angle_poly)
        return x_delta + loc[0], y_delta + loc[1]

    def getStartLocation(self):
        x_min = np.amin(self.polygon[:, 0])
        x_max = np.amax(self.polygon[:, 0])
        y_min = np.amin(self.polygon[:, 1])
        y_max = np.amax(self.polygon[:, 1])
        path_polygon = mplPath.Path(self.polygon)
        while True:
            x_random = np.random.uniform(x_min, x_max)
            y_random = np.random.uniform(y_min, y_max)
            if path_polygon.contains_point((x_random, y_random)):
                break
        print("The generated random starting location is: ")
        print([x_random, y_random])
        self.loc_start = [x_random, y_random]

    def getGridPoly(self):
        '''
        get the polygon grid discretisation
        '''
        self.getStartLocation()
        x_new, y_new = self.getNewLocations(self.loc_start)
        start_node = []
        for i in range(len(self.angle_poly)):
            if self.polygon_path.contains_point((x_new[i], y_new[i])):
                start_node.append([x_new[i], y_new[i]])
                self.grid_poly.append([x_new[i], y_new[i]])
                self.counter_grid = self.counter_grid + 1

        if self.debug:
            plt.figure(figsize=(10, 10))
            # temp1 = np.array(self.grid_poly)
            # plt.plot(temp1[:, 1], temp1[:, 0], 'k.')
            plt.plot(self.loc_start[1], self.loc_start[0], 'rx', label="Starting location")
            plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'k-.', label="Polygon")
            plt.xlabel("Distance along x direction [m]")
            plt.ylabel("Distance along y direction [m]")
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Dynamic grid generation")
            plt.savefig(self.figpath + "/I_{:04d}.pdf".format(self.counter_plot), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close("all")

            self.counter_plot = self.counter_plot + 1

            print(self.counter_grid)
            plt.figure(figsize=(10, 10))
            temp1 = np.array(self.grid_poly)
            plt.plot(temp1[:, 1], temp1[:, 0], 'b.')
            plt.plot(self.loc_start[1], self.loc_start[0], 'rx', label="Current location")
            plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'k-.', label = "Polygon")
            plt.xlabel("Distance along x direction [m]")
            plt.ylabel("Distance along y direction [m]")
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Dynamic grid generation")

            plt.savefig(self.figpath + "/I_{:04d}.pdf".format(self.counter_plot), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close("all")
        WaypointNode_start = WaypointNode(len(start_node), start_node, self.loc_start)
        Allwaypoints = self.getAllWaypoints(WaypointNode_start)
        self.grid_poly = np.array(self.grid_poly)
        if len(self.grid_poly) > self.pointsPr:
            print("{:d} waypoints are generated, only {:d} waypoints are selected!".format(len(self.grid_poly),
                                                                                           self.pointsPr))
            self.grid_poly = self.grid_poly[:self.pointsPr, :]
        else:
            print("{:d} waypoints are generated, all are selected!".format(len(self.grid_poly)))
        print("Grid: ", self.grid_poly.shape)

    def getAllWaypoints(self, waypoint_node):
        if self.counter_grid > self.pointsPr:  # stopping criterion to end the recursion
            return WaypointNode(0, [], waypoint_node.waypoint_loc)
        for i in range(waypoint_node.subwaypoint_len):  # loop through all the subnodes
            subsubwaypoint = []
            length_new = 0
            x_subsubwaypoint, y_subsubwaypoint = self.getNewLocations(
                waypoint_node.subwaypoint_loc[i])  # generate candidates location
            for j in range(len(self.angle_poly)):
                if self.polygon_path.contains_point((x_subsubwaypoint[j], y_subsubwaypoint[j])):
                    testRevisit = self.revisit([x_subsubwaypoint[j], y_subsubwaypoint[j]])
                    if not testRevisit[0]:
                        subsubwaypoint.append([x_subsubwaypoint[j], y_subsubwaypoint[j]])
                        self.grid_poly.append([x_subsubwaypoint[j], y_subsubwaypoint[j]])
                        self.counter_grid = self.counter_grid + 1
                        length_new = length_new + 1
            if len(subsubwaypoint) > 0:
                if self.debug:
                    self.counter_plot = self.counter_plot + 1
                    print(self.counter_grid)
                    plt.figure(figsize=(10, 10))
                    temp1 = np.array(self.grid_poly)
                    plt.plot(temp1[:, 1], temp1[:, 0], 'b.')
                    plt.plot(temp1[-length_new:][:, 1], temp1[-length_new:][:, 0], 'g*', label = "New candidate location")
                    plt.plot(waypoint_node.subwaypoint_loc[i][1], waypoint_node.subwaypoint_loc[i][0], 'rx', label = "Current location")
                    plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'k-.', label = "Polygon")
                    plt.xlabel("Distance along x direction [m]")
                    plt.ylabel("Distance along y direction [m]")
                    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.title("Dynamic grid generation")
                    plt.savefig(self.figpath + "/I_{:04d}.pdf".format(self.counter_plot), bbox_extra_artists=(lgd,),
                                bbox_inches='tight')
                    plt.close("all")
                Subwaypoint = WaypointNode(len(subsubwaypoint), subsubwaypoint, waypoint_node.subwaypoint_loc[i])
                self.getAllWaypoints(Subwaypoint)
            else:
                return WaypointNode(0, [], waypoint_node.subwaypoint_loc[i])
        return WaypointNode(0, [], waypoint_node.waypoint_loc)

    def getPolygonArea(self):
        area = 0
        prev = self.polygon[-2]
        for i in range(self.polygon.shape[0] - 1):
            now = self.polygon[i]
            xnow, ynow = now[0], now[1]
            xpre, ypre = prev[0], prev[1]
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2
        print("Area: ", self.PolyArea / 1e6, " km2")
        if self.voiceCtrl:
            os.system("say Area is: {:.1f} squared kilometers".format(self.PolyArea / 1e6))

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180


if __name__ == "__main__":
    polygon = np.array([[0, 0],
                        [10, 0],
                        [10, 10],
                        [0, 10],
                        [0, 0]])
    # polygon = np.array([[0, 5],
    #                     [1, 4],
    #                     [2, 3],
    #                     [4, 2],
    #                     [6, 3],
    #                     [8, 6],
    #                     [9, 5],
    #                     [7, 3],
    #                     [4, 0],
    #                     [1, 2],
    #                     [0, 4],
    #                     [0, 5]]) * 10
    # x = np.arange(10)
    # y = np.arange(10)
    # xx, yy = np.meshgrid(x, y)
    # plt.plot(xx, yy, '.')
    # plt.show()
    np.random.seed(1)
    grid = GridPoly(polygon = polygon)



