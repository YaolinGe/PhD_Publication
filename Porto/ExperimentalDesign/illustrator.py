import numpy as np
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import mvn
np.random.seed(2021)

class Illustrator:

    N1 = 25
    N2 = 25
    N3 = 5
    N = N1 * N2 * N3
    x = np.arange(N1)
    y = np.arange(N2)
    z = np.arange(N3)
    sigma = np.sqrt(.0002)
    eta = 4.5 / 5
    tau = np.sqrt(.04)
    R = tau ** 2
    threshold = .15 # threshold

    def __init__(self):
        print("Hello, here comes the illustrator!")
        self.get_grid_values()
        self.get_distance_matrix()
        self.get_true_mean()
        # self.find_path()
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
        print("Hello")
        self.path = []
        xs, ys, zs = [24, 10, 0] # starting location
        x, y, z = xs, ys, zs
        xp, yp, zp = x, y, z
        mc = self.val
        Sigc = self.CovM
        No_steps = 15
        F = np.zeros([1, self.N])
        ind_n = self.ravel_index([x, y, z], self.N1, self.N2, self.N3)
        F[0, ind_n] = True
        y_sampled = F @ self.vol
        self.path.append([x, y, z, y_sampled[0]])
        for i in range(No_steps):
            print(i)
            xc, yc, zc = self.find_candidates_loc(x, y, z, self.N1, self.N2, self.N3)
            xn, yn, zn = self.find_next_EIBV_1D(xc, yc, zc, x, y, z, xp, yp, zp, self.N1, self.N2, self.N3, Sigc,
                                                mc, self.tau, self.threshold)
            print(xn, yn, zn)
            ind_n = self.ravel_index([xn, yn, zn], self.N1, self.N2, self.N3)
            # print(ind_n)
            F = np.zeros([1, self.N])
            F[0, ind_n] = True
            y_sampled = F @ self.vol
            mc, Sigc = self.GPupd(mc, Sigc, self.R, F, y_sampled)

            xp, yp, zp = x, y, z
            x, y, z = xn, yn, zn
            self.path.append([x, y, z, y_sampled[0]])
        self.path = np.array(self.path)
        np.savetxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/ExperimentalDesign/path.txt",
                   self.path, delimiter=", ")
        print("Path is saved successfully!")

    def GPupd(self, mu, Sig, R, F, y_sampled):
        C = F @ Sig @ F.T + R
        mu_p = mu + Sig @ F.T @ np.linalg.solve(C, (y_sampled - F @ mu))
        Sigma_p = Sig - Sig @ F.T @ np.linalg.solve(C, F @ Sig)
        return mu_p, Sigma_p

    def ravel_index(self, loc, n1, n2, n3):
        x, y, z = loc
        ind = int(z * n1 * n2 + y * n1 + x)
        return ind

    def unravel_index(self, ind, n1, n2, n3):
        zind = np.floor(ind / (n1 * n2))
        residual = ind - zind * (n1 * n2)
        yind = np.floor(residual / n1)
        xind = residual - yind * n1
        loc = [int(xind), int(yind), int(zind)]
        return loc

    def EIBV_1D(self, threshold, mu, Sig, F, R):
        Sigxi = Sig @ F.T @ np.linalg.solve(F @ Sig @ F.T + R, F @ Sig)
        V = Sig - Sigxi
        sa2 = np.diag(V).reshape(-1, 1)  # the corresponding variance term for each location
        IntA = 0.0
        for i in range(len(mu)):
            sn2 = sa2[i]
            m = mu[i]
            IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
        return IntA

    def find_candidates_loc(self, x_ind, y_ind, z_ind, N1, N2, N3):
        x_ind_l = [x_ind - 1 if x_ind > 0 else x_ind]
        x_ind_u = [x_ind + 1 if x_ind < N1 - 1 else x_ind]
        y_ind_l = [y_ind - 1 if y_ind > 0 else y_ind]
        y_ind_u = [y_ind + 1 if y_ind < N2 - 1 else y_ind]
        z_ind_l = [z_ind - 1 if z_ind > 0 else z_ind]
        z_ind_u = [z_ind + 1 if z_ind < N3 - 1 else z_ind]

        x_ind_v = np.unique(np.vstack((x_ind_l, x_ind, x_ind_u)))
        y_ind_v = np.unique(np.vstack((y_ind_l, y_ind, y_ind_u)))
        z_ind_v = np.unique(np.vstack((z_ind_l, z_ind, z_ind_u)))

        x_ind, y_ind, z_ind = np.meshgrid(x_ind_v, y_ind_v, z_ind_v)
        return x_ind.reshape(-1, 1), y_ind.reshape(-1, 1), z_ind.reshape(-1, 1)

    def find_next_EIBV_1D(self, x_cand, y_cand, z_cand, x_now, y_now, z_now,
                          x_pre, y_pre, z_pre, N1, N2, N3, Sig, mu, tau, Threshold):

        id = []
        dx1 = x_now - x_pre
        dy1 = y_now - y_pre
        dz1 = z_now - z_pre
        vec1 = np.array([dx1, dy1, dz1])
        for i in x_cand:
            for j in y_cand:
                for z in z_cand:
                    if i == x_now and j == y_now and z == z_now:
                        continue
                    dx2 = i - x_now
                    dy2 = j - y_now
                    dz2 = z - z_now
                    vec2 = np.array([dx2, dy2, dz2])
                    if np.dot(vec1, vec2) >= 0:
                        id.append(self.ravel_index([i, j, z], N1, N2, N3))
                    else:
                        continue
        id = np.unique(np.array(id))

        M = len(id)
        noise = tau ** 2
        R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector
        N = N1 * N2 * N3
        eibv = []
        for k in range(M):
            F = np.zeros([1, N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(Threshold, mu, Sig, F, R))
        ind_desired = np.argmin(np.array(eibv))
        x_next, y_next, z_next = self.unravel_index(id[ind_desired], N1, N2, N3)

        return x_next, y_next, z_next

    def plot_illustrator(self):
        try:
            self.path = np.loadtxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/ExperimentalDesign/path.txt", delimiter = ", ")
        except:
            print("No ")

        fig = go.Figure(data=[go.Scatter3d(
            x=self.path[:, 0],
            y=self.path[:, 1],
            z=self.path[:, 2],
            # mode='markers + lines',
            mode='lines',
            # marker=dict(
            #     size=5,
            #     color = "orange",
            #     showscale = False,
            # ),
            line = dict(
                color = "yellow",
                width = 3,
                showscale = False,

        )
        )])
        fig.add_trace(go.Scatter3d(
            x=self.path[:, 0],
            y=self.path[:, 1],
            z=self.path[:, 2],
            mode='markers',
            marker=dict(
                size = self.path[:, 3] * 100,
                color = self.path[:, 3] * 100 + 17,
                # colorscale = "RdBu",
                coloraxis = "coloraxis",
                showscale = True,
                reversescale = True,
            ),
        ))
        fig.update_coloraxes(colorscale="RdBu", colorbar=dict(lenmode='fraction', len=0.5, thickness=20,
                                                                tickfont=dict(size=18, family="Times New Roman"),
                                                                title="Salinity",
                                                                titlefont=dict(size=18, family="Times New Roman")),
                             reversescale = True, )
        fig.update_layout(coloraxis_colorbar_x=.75)
        fig.add_trace(go.Volume(
            x=self.grid[:, 0].flatten(),
            y=self.grid[:, 1].flatten(),
            z=self.grid[:, 2].flatten(),
            value=self.vol.flatten(),
            # isomin=,
            isomax=self.threshold - .02,
            opacity=0.1,
            # coloraxis="coloraxis",
            showscale = False,
            # surface_count=5,
            ))
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)
        fig.update_scenes(xaxis_visible=True,
                          yaxis_visible=True,
                          zaxis_visible=True )
        # fig.update_layout(scene_xaxis_showticklabels=True,
        #                   scene_yaxis_showticklabels=True,
        #                   scene_zaxis_showticklabels=True)
        # fig.update_coloraxes(colorscale = "GnBu")

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                # xaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=True),
                # yaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=True),
                # zaxis=dict(autorange='reversed', nticks=4, range=[0, 5], showticklabels=True),

                xaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=False),
                yaxis=dict(autorange='reversed', nticks=4, range=[0, 10], showticklabels=False),
                zaxis=dict(autorange='reversed', nticks=4, range=[0, 5], showticklabels=False),
                xaxis_title=dict(text="Longitude", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Latitude", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Depth", font=dict(size=18, family="Times New Roman")),
        annotations = [
            dict(
                x=self.path[0, 0],
                y=self.path[0, 1],
                z=self.path[0, 2],
                text="Starting location",
                textangle=0,
                ax=-50,
                ay=-70,
                font=dict(
                    color="black",
                    size=18,
                    family="Times New Roman"
                ),
                arrowcolor="black",
                arrowsize=3,
                arrowwidth=1,
                arrowhead=1),
            dict(
                x=20.5,
                y=18.5,
                z=0,
                text="Sampling path",
                textangle=0,
                ax=-50,
                ay=-70,
                font=dict(
                    color="yellow",
                    size=18,
                    family="Times New Roman"
                ),
                arrowcolor="yellow",
                arrowsize=3,
                arrowwidth=1,
                arrowhead=1),
            ],
            ),
        )
        fig.update_layout(scene_aspectmode='manual',
                          scene_aspectratio=dict(x=1, y=1, z=.5))
        plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/illustrator/illu.html", auto_open=True)


if __name__ == "__main__":
    a = Illustrator()


