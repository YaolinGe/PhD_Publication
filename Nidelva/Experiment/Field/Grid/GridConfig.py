

class GridConfig:

    def __init__(self, pivot, angle_rotation, nx, ny, nz, xlim, ylim, zlim):
        self.lat_pivot, self.lon_pivot = pivot
        self.angle_rotation = angle_rotation
        self.number_of_points_x = nx
        self.number_of_points_y = ny
        self.number_of_points_z = nz
        self.number_of_points_total_grid = nx * ny * nz
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
