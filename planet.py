import numpy as np

class Planet:

    def __init__(self, planetID):

        if planetID == "NEPTUNE":
            self.GM = 6.8365299E15
            self.a0 = 299.36 * np.pi/180.0
            self.d0 = 43.36 * np.pi/180.0

        if planetID == "URANUS":
            self.GM = 5.793939E15
            self.a0 = 257.311 * np.pi/180.0
            self.d0 = -15.175 * np.pi/180.0

        if planetID == "SATURN":
            self.GM = 3.7931187E16
            self.a0 = 39.4827 * np.pi / 180.0
            self.d0 = 83.4279 * np.pi / 180.0

