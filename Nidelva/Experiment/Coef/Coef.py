
import numpy as np


class Coef:

    def __init__(self):
        coef_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Coef/"
        self.beta0 = np.loadtxt(coef_path + "beta0.txt", delimiter=",")
        self.beta1 = np.loadtxt(coef_path + "beta1.txt", delimiter=",")
        pass
