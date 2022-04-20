# SOURCE FILENAME : probe_trajectory.py
# AUTHOR          : Athul Pradeepkumar Girija, apradee@purdue.edu
# DATE MODIFIED   : 04/19/2022, 19:31 MT
# REMARKS         : Compute the probe arrival trajectory
#                   following Kyle's dissertation Chapter 2

import numpy as np
from numpy import linalg as LA
from planet import Planet

class Approach:

    def __init__(self, arrivalPlanet, v_inf_vec_icrf, rp, psi):
        self.planetObj = Planet(arrivalPlanet)
        self.a0 = self.planetObj.a0
        self.d0 = self.planetObj.d0

        self.v_inf_vec_icrf = v_inf_vec_icrf
        self.rp = rp
        self.psi = psi

        self.v_inf_mag = LA.norm(self.v_inf_vec_icrf)

        self.a = -self.planetObj.GM/((self.v_inf_mag*1e3)**2)
        self.e = 1 - self.rp/self.a
        self.beta = np.arccos(1.0/self.e)

        self.v_inf_vec_bi = self.ICRF_to_BI(self.v_inf_vec_icrf)

        self.phi_1 = np.arctan(self.v_inf_vec_bi[1]/self.v_inf_vec_bi[0])

        self.v_inf_vec_bi_prime = (np.matmul(self.R3(self.phi_1), self.v_inf_vec_bi.T)).T

        self.phi_2 = np.arctan(self.v_inf_vec_bi_prime[0]/self.v_inf_vec_bi[2])


    def R1(self, theta):
        return np.array([[1, 0, 0],
                        [0, np.cos(theta), np.sin(theta)],
                        [0, -np.sin(theta), np.cos(theta)]])

    def R3(self, theta):
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def ICRF_to_BI(self, X_ICRF):

        return (np.matmul(self.R1(np.pi/2 - self.d0),
                np.matmul(self.R3(np.pi/2 + self.a0), X_ICRF.T))).T


class Test_Approach_Neptune:

    approach = Approach("NEPTUNE",
                        v_inf_vec_icrf=np.array([17.78952518,  8.62038536,  3.15801163]),
                        rp=(24764+400)*1e3, psi=3*np.pi/2)

    def test_R1_0(self):

        assert (self.approach.R1(0) == np.array([[1,0,0],
                                               [0, np.cos(0), np.sin(0)],
                                               [0, -np.sin(0), np.cos(0)]])).all()
    def test_R1_90(self):

        assert (self.approach.R1(np.pi/2) == np.array([[1, 0, 0],
                                                 [0, np.cos(np.pi/2), np.sin(np.pi/2)],
                                                 [0, -np.sin(np.pi/2), np.cos(np.pi/2)]])).all()


    def test_R3_0(self):

        assert (self.approach.R3(0) == np.array([[np.cos(0), np.sin(0), 0],
                                                [-np.sin(0), np.cos(0), 0],
                                                [0, 0, 1]])).all()

    def test_R3_90(self):
        assert (self.approach.R3(np.pi/2) == np.array([[np.cos(np.pi/2), np.sin(np.pi/2), 0],
                                                        [-np.sin(np.pi/2), np.cos(np.pi/2), 0],
                                                        [0, 0, 1]])).all()

    def test_ICRF_to_BI_unit_vec(self):

        delta =  abs(np.linalg.norm(self.approach.ICRF_to_BI(np.array([1, 0, 0])))) - 1
        assert delta < 1e-8

    def test_ICRF_to_BI_123(self):

        delta = abs(np.linalg.norm(self.approach.ICRF_to_BI(np.array([1, 2, 3])))) - np.sqrt(14)
        assert delta < 1e-8

    def test_a(self):
        assert ((abs(self.approach.a) - abs(-17059283.6903))/ abs(-17059283.6903)) < 1e-6

    def test_e(self):
        assert (abs(self.approach.e - 2.47509) / 2.47509) < 1e-4

    def test_beta(self):
        assert (abs(self.approach.beta - 1.15488) / 1.15488) < 1e-4

    def test_v_inf_vec_bi(self):
        delta =  abs(np.linalg.norm(self.approach.v_inf_vec_bi) - self.approach.v_inf_mag)
        assert delta < 1e-8

    def test_phi_1(self):
        delta = abs(self.approach.phi_1) - abs(0.0741621)
        assert delta < 1e-4

    def test_v_inf_vec_bi_prime(self):
        delta = abs(np.linalg.norm(self.approach.v_inf_vec_bi_prime) - self.approach.v_inf_mag)
        assert delta < 1e-8

    def test_phi_2(self):
        delta = abs(self.approach.phi_2) - abs(1.417984)
        assert delta < 1e-4

    def test_phi_2_analytic(self):
        phi_2_analytic = np.arctan((self.approach.v_inf_vec_bi[0]*np.cos(self.approach.phi_1) +\
                                    self.approach.v_inf_vec_bi[1]*np.cos(self.approach.phi_1)) /\
                                    self.approach.v_inf_vec_bi[2])

        assert abs(self.approach.phi_2 - phi_2_analytic) < 1e-2

