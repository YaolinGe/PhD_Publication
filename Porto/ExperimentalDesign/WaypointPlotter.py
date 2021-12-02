import plotly
import numpy as np
import plotly.graph_objects as go

class WaypointGraphPlotter:


    def __init__(self):
        print("Hello world")
        self.get_grid()
        self.find_path()
        self.plot_waypoint_3d()

    def get_grid(self):
        x = np.arange(10)
        y = np.arange(10)
        z = np.arange(5)
        xx, yy, zz = np.meshgrid(x, y, z)
        self.X = xx.flatten()
        self.Y = yy.flatten()
        self.Z = zz.flatten()

    def find_path(self):
        self.path = []
        self.path.append([0, 0, 0])
        self.path.append([1, 1, 0])
        self.path.append([2, 2, 1])
        self.path.append([3, 2, 1])
        self.path.append([4, 3, 1])
        self.path.append([4, 4, 2])
        self.path = np.array(self.path)

        v1 = np.array([0, 1, 1])
        self.x_n = np.array([3, 4, 5])
        self.y_n = np.array([3, 4, 5])
        self.z_n = np.array([1, 2, 3])
        self.xxn, self.yyn, self.zzn = np.meshgrid(self.x_n, self.y_n, self.z_n)
        self.Xn = self.xxn.flatten()
        self.Yn = self.yyn.flatten()
        self.Zn = self.zzn.flatten()
        self.path_n = []
        for i in range(len(self.Xn)):
            if self.Xn[i] == 4 and self.Yn[i] == 4 and self.Zn[i] == 2:
                pass
            else:
                v2 = np.array([self.Xn[i] - 4, self.Yn[i] - 4, self.Zn[i] - 2])
                print("v: ", np.dot(v1, v2))
                if np.dot(v1, v2) >= 0:
                    self.path_n.append([self.Xn[i], self.Yn[i], self.Zn[i]])
                else:
                    print("wrong")
        self.path_n = np.array(self.path_n)


    def plot_waypoint_3d(self):
        # Here comes the path planning

        fig = go.Figure(data=[go.Scatter3d(
            x=self.path[:, 0],
            y=self.path[:, 1],
            z=self.path[:, 2],
            mode='markers + lines',
            marker=dict(
                size=5,
                color = "black",
                showscale = False,
            ),
            line = dict(
                color = "orange",
                width = 5,
                showscale = False,

        )
        )])

        fig.add_trace(go.Scatter3d(
            x=self.path_n[:, 0],
            y=self.path_n[:, 1],
            z=self.path_n[:, 2],
            mode='markers',
            marker=dict(
                size=1 / np.sum(self.path_n, axis = 1) ** 2 * 1000,
                color = 1 / np.sum(self.path_n, axis = 1) ** 2 * 10,
                coloraxis = "coloraxis",
                # reversescale = True,
                # colorscale = "rainbow",
                # showscale = False,

            ),

        ))

        fig.add_trace(go.Scatter3d(
            x=[4],
            y=[4],
            z=[2],
            mode='markers',
            marker=dict(
                size=5,
                color = "blue",
                showscale = False,
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=[5],
            y=[5],
            z=[3],
            mode='markers',
            marker=dict(
                size=3,
                color = "green",
                showscale = False,
            ),
        ))

        fig.add_trace(go.Cone(
            x=[4.15], y=[4.15], z=[2.15], u=[1], v=[1], w=[1],
            showscale=False
        ))

        # tight layout
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene = dict(
            xaxis = dict(nticks=4, range=[0, 10], showticklabels=False),
            yaxis = dict(nticks=4, range=[0, 10], showticklabels=False),
            zaxis=dict(nticks=4, range=[0, 5], showticklabels=False),
            xaxis_title=dict(text = "East", font = dict(size = 18, family = "Times New Roman")),
            yaxis_title=dict(text = "North", font = dict(size = 18, family = "Times New Roman")),
            zaxis_title=dict(text = "Depth", font = dict(size = 18, family = "Times New Roman")),
            # font=dict(
            #     family="Times New Roman",
            #     size=18,
            # # color="RebeccaPurple"
            # ),
            annotations=[
                dict(
                    x=0,
                    y=0,
                    z=0,
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
                    x=4,
                    y=4,
                    z=2,
                    text="Current location",
                    textangle=0,
                    ax=70,
                    ay=90,
                    font=dict(
                        color="blue",
                        size=18,
                        family="Times New Roman"
                    ),
                    arrowcolor="blue",
                    arrowsize=3,
                    arrowwidth=1,
                    arrowhead=1),
                dict(
                    x=5,
                    y=5,
                    z=3,
                    text="Next desired location",
                    textangle=0,
                    ax=50,
                    ay=-70,
                    font=dict(
                        color="green",
                        size=18,
                        family = "Times New Roman"
                    ),
                    arrowcolor="green",
                    arrowsize=3,
                    arrowwidth=1,
                    arrowhead=1),
                ],
            ),
            # title="Plot Title",

            # legend_title="Legend Title",

          scene_aspectmode='manual',
          scene_aspectratio=dict(x=1, y=1, z=.5)
        )
        fig.update_coloraxes(colorscale = "Brwnyl", colorbar=dict(lenmode='fraction', len=0.5, thickness=20, tickfont = dict(size = 18, family = "Times New Roman"), title = "EIBV", titlefont = dict(size = 18, family = "Times New Roman")))
        fig.update_layout(coloraxis_colorbar_x=0.75)
        # fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/waypoint/waypoint.pdf", width=1980, height=1080,)
        plotly.offline.plot(fig, "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/waypoint/waypoint.html", auto_open=True)
        print("Now")


if __name__ == "__main__":
    a = WaypointGraphPlotter()
    print("End")
