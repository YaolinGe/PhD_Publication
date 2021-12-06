import numpy as np
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

np.random.seed(2021)
class Illustrator:

    x = np.arange(25)
    y = np.arange(25)
    z = np.arange(5)
    sigma = np.sqrt(.0002)
    eta = 4.5 / 5
    tau = np.sqrt(.04)

    def __init__(self):
        print("Hello, here comes the illustrator!")
        self.get_grid_values()
        self.get_distance_matrix()
        self.get_true_mean()
        self.find_path()
        self.plot_illustrator()

    def get_grid_values(self):
        self.val = []
        self.grid = []
        for i in range(len(self.z)):
            for j in range(len(self.x)):
                for k in range(len(self.y)):
                    temp = 3 * (self.x[j] - 12) ** 2 + self.y[k] ** 2 + 200 * self.z[i] ** 2
                    self.val.append(temp)
                    self.grid.append([self.x[j], self.y[k], self.z[i]])

        self.grid = np.array(self.grid)
        self.val = np.array(self.val)
        self.val = self.val / np.amax(self.val)

    def get_distance_matrix(self):
        x = self.grid[:, 0].reshape(-1, 1)
        y = self.grid[:, 1].reshape(-1, 1)
        z = self.grid[:, 2].reshape(-1, 1)

        dist_x = x @ np.ones([1, x.shape[0]]) - np.ones([x.shape[0], 1]) @ x.T
        dist_y = y @ np.ones([1, y.shape[0]]) - np.ones([y.shape[0], 1]) @ y.T
        dist_z = z @ np.ones([1, z.shape[0]]) - np.ones([z.shape[0], 1]) @ z.T
        distanceM = np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

        self.CovM = self.sigma ** 2 * (1 + self.eta * distanceM) * np.exp(-self.eta * distanceM)

    def get_true_mean(self):
        self.vol = self.val + np.linalg.cholesky(self.CovM) @ np.random.randn(len(self.val))


    def find_path(self):
        # self.path = []
        # self.path.append([13, 23, 0])
        # self.path.append([16, 21, 0])
        # self.path.append([22, 16, 1])
        # self.path.append([21, 15, 1])
        # self.path.append([21, 14, 1])
        # # self.path.append([, 19, 1])
        # # self.path.append([4, 4, 2])
        # self.path = np.array(self.path)

    def plot_illustrator(self):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.path[:, 0],
            y=self.path[:, 1],
            z=self.path[:, 2],
            mode='markers + lines',
            marker=dict(
                size=5,
                color = "orange",
                showscale = False,
            ),
            line = dict(
                color = "orange",
                width = 5,
                showscale = False,

        )
        )])
        fig.add_trace(go.Volume(
            x=self.grid[:, 0].flatten(),
            y=self.grid[:, 1].flatten(),
            z=self.grid[:, 2].flatten(),
            value=self.vol.flatten(),
            # isomin=,
            isomax=.12,
            opacity=0.1,
            coloraxis="coloraxis",
            # surface_count=5,
            ))
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        # fig.update_scenes(xaxis_visible=False,
        #                   yaxis_visible=False,
        #                   zaxis_visible=False )
        # fig.update_layout(scene_xaxis_showticklabels=True,
        #                   scene_yaxis_showticklabels=True,
        #                   scene_zaxis_showticklabels=True)
        fig.update_coloraxes(colorscale = "GnBu")
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=True),
                yaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=True),
                zaxis=dict(autorange='reversed', nticks=4, range=[0, 5], showticklabels=True),

                # xaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=False),
                # yaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=False),
                # zaxis=dict(autorange='reversed', nticks=4, range=[0, 5], showticklabels=False),
                xaxis_title=dict(text="Longitude", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Latitude", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Depth", font=dict(size=18, family="Times New Roman")),
            ))
        fig.update_layout(scene_aspectmode='manual',
                          scene_aspectratio=dict(x=1, y=1, z=.5))
        plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/illustrator/illu.html", auto_open=True)


if __name__ == "__main__":
    a = Illustrator()


