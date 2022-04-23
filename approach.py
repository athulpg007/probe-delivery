# SOURCE FILENAME : approach.py
# AUTHOR          : Athul Pradeepkumar Girija, apradee@purdue.edu
# DATE MODIFIED   : 04/20/2022, 21:29 MT
# REMARKS         : Compute the probe arrival trajectory
#                   following Kyle's dissertation Chapter 2

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from planet import Planet


class Approach:

    def __init__(self, arrivalPlanet, v_inf_vec_icrf, rp, psi, h_EI):
        self.planetObj = Planet(arrivalPlanet)
        self.a0 = self.planetObj.a0
        self.d0 = self.planetObj.d0

        self.v_inf_vec_icrf = v_inf_vec_icrf
        self.rp = rp
        self.psi = psi
        self.h_EI = h_EI
        self.r_EI = self.planetObj.RP + self.h_EI

        self.v_inf_mag = LA.norm(self.v_inf_vec_icrf)

        self.a = -self.planetObj.GM/((self.v_inf_mag*1e3)**2)
        self.e = 1 - self.rp/self.a
        self.beta = np.arccos(1.0/self.e)

        self.v_inf_vec_bi = self.ICRF_to_BI(self.v_inf_vec_icrf)

        self.v_inf_vec_bi_unit = self.v_inf_vec_bi / LA.linalg.norm(self.v_inf_vec_bi)

        self.phi_1 = np.arctan(self.v_inf_vec_bi[1]/self.v_inf_vec_bi[0])

        self.v_inf_vec_bi_prime = (np.matmul(self.R3(self.phi_1), self.v_inf_vec_bi.T)).T

        self.phi_2 = np.arctan(self.v_inf_vec_bi_prime[0]/self.v_inf_vec_bi[2])

        self.rp_vec_bi_x_unit = np.cos(self.phi_1)*\
                                (np.sin(self.beta)*np.cos(self.psi)*np.cos(self.phi_2) + np.cos(self.beta)*np.sin(self.phi_2)) -\
                                np.sin(self.phi_1)*np.sin(self.beta)*np.sin(self.psi)

        self.rp_vec_bi_y_unit = np.sin(self.phi_1)*\
                                (np.sin(self.beta)*np.cos(self.psi)*np.cos(self.phi_2) + np.cos(self.beta)*np.sin(self.phi_2)) + \
                                np.cos(self.phi_1)*np.sin(self.beta)*np.sin(self.psi)

        self.rp_vec_bi_z_unit = np.cos(self.beta)*np.cos(self.phi_2) - np.sin(self.beta)*np.cos(self.psi)*np.sin(self.phi_2)

        self.rp_vec_bi_unit = np.array([self.rp_vec_bi_x_unit,
                                        self.rp_vec_bi_y_unit,
                                        self.rp_vec_bi_z_unit])

        self.rp_vec_bi = self.rp*self.rp_vec_bi_unit

        self.rp_vec_bi_dprime = self.rp*np.array([np.sin(self.beta)*np.cos(self.psi),
                                                  np.sin(self.beta)*np.sin(self.psi),
                                                  np.cos(self.beta)])

        self.e_vec_bi = self.e*self.rp_vec_bi_unit

        self.e_vec_bi_unit = self.e_vec_bi / LA.linalg.norm(self.e_vec_bi)

        self.h = np.sqrt(self.a*self.planetObj.GM*(1-self.e**2))

        self.h_vec_bi = np.cross(self.rp_vec_bi, self.v_inf_vec_bi)

        self.h_vec_bi_unit = self.h_vec_bi / LA.linalg.norm(self.h_vec_bi)

        self.i = np.arccos(self.h_vec_bi_unit[2])

        self.N_vec_bi_unit = np.cross(np.array([0, 0, 1]), self.h_vec_bi_unit)

        if self.N_vec_bi_unit[1] >= 0:
            self.OMEGA = np.arccos(self.N_vec_bi_unit[0])
        else:
            self.OMEGA = 2*np.pi - np.arccos(self.N_vec_bi_unit[0])

        if self.e_vec_bi_unit[2] >= 0:
            self.omega = np.arccos(np.dot(self.N_vec_bi_unit, self.e_vec_bi_unit))
        else:
            self.omega = 2*np.pi - np.arccos(np.dot(self.N_vec_bi_unit, self.e_vec_bi_unit))

        self.N_ref_bi_vec_unit = np.array([0, 0, 1])
        self.S_vec_bi_unit = self.v_inf_vec_bi_unit
        self.T_vec_bi_unit = np.cross(self.S_vec_bi_unit, self.N_ref_bi_vec_unit)
        self.R_vec_bi_unit = np.cross(self.S_vec_bi_unit, self.T_vec_bi_unit)
        self.B_vec_bi_unit = np.cross(self.S_vec_bi_unit, self.h_vec_bi_unit)

        self.b_plane_angle_theta = self.psi + np.pi/2

        self.theta_star_entry = -1*np.arccos(((self.h**2/(self.planetObj.GM * self.r_EI)) - 1)*(1.0/self.e))

        self.r_vec_entry_bi = self.pos_vec_bi(self.theta_star_entry)
        self.r_vec_entry_bi_unit = self.r_vec_entry_bi / LA.linalg.norm(self.r_vec_entry_bi)
        self.r_vec_entry_bi_mag = LA.linalg.norm(self.r_vec_entry_bi)

        self.v_entry_inertial_mag = np.sqrt((self.v_inf_mag*1e3)**2 + 2*self.planetObj.GM/self.r_EI)

        self.gamma_entry_inertial = -1*np.arccos(self.h/(self.r_EI*self.v_entry_inertial_mag))

        self.v_vec_entry_bi = self.vel_vec_bi(self.theta_star_entry)

        self.latitude_entry = np.arcsin(self.r_vec_entry_bi_unit[2])
        self.longitude_entry = np.arctan(self.r_vec_entry_bi_unit[1]/self.r_vec_entry_bi_unit[0])

        self.v_entry_atm_bi = np.array([ self.v_vec_entry_bi[0] - \
                                         self.r_vec_entry_bi_mag * self.planetObj.OMEGA*np.cos(self.latitude_entry)*\
                                         (-np.sin(self.longitude_entry)),
                                         self.v_vec_entry_bi[1] - \
                                         self.r_vec_entry_bi_mag * self.planetObj.OMEGA*np.cos(self.latitude_entry)*\
                                         (np.cos(self.longitude_entry)),
                                         self.v_vec_entry_bi[2]
                                         ])

        self.v_entry_atm_bi_mag = LA.linalg.norm(self.v_entry_atm_bi)





    def R1(self, theta):
        return np.array([[1, 0, 0],
                        [0, np.cos(theta), np.sin(theta)],
                        [0, -np.sin(theta), np.cos(theta)]])

    def R2(self, theta):
        return np.array([[np.cos(theta), 0,  -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0,  np.cos(theta)]])

    def R3(self, theta):
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def ICRF_to_BI(self, X_ICRF):

        return (np.matmul(self.R1(np.pi/2 - self.d0),
                np.matmul(self.R3(np.pi/2 + self.a0), X_ICRF.T))).T


    def pos_vec_bi(self, theta_star):

        r = (self.h**2 / self.planetObj.GM) / (1 + self.e*np.cos(theta_star))
        theta = theta_star + self.omega

        rx_unit = np.cos(self.OMEGA)*np.cos(theta) - np.sin(self.OMEGA)*np.cos(self.i)*np.sin(theta)
        ry_unit = np.sin(self.OMEGA)*np.cos(theta) + np.cos(self.OMEGA)*np.cos(self.i)*np.sin(theta)
        rz_unit = np.sin(self.i)*np.sin(theta)

        pos_vec_bi = r*np.array([rx_unit, ry_unit, rz_unit])
        return pos_vec_bi

    def r_mag_bi(self, theta_star):
        r_mag_bi = (self.h**2 / self.planetObj.GM) / (1 + self.e*np.cos(theta_star))
        return r_mag_bi

    def vel_vec_bi(self, theta_star):

        r_mag_bi = self.r_mag_bi(theta_star)
        v_mag_bi = np.sqrt((self.v_inf_mag*1e3)**2 + 2*self.planetObj.GM/r_mag_bi)
        gamma = -1*np.arccos(self.h/(r_mag_bi*v_mag_bi))

        vr = v_mag_bi*np.sin(gamma)
        vt = v_mag_bi*np.cos(gamma)
        theta = theta_star + self.omega

        vx = vr*( np.cos(theta)*np.cos(self.OMEGA) - np.sin(theta)*np.cos(self.i)*np.sin(self.OMEGA)) +\
             vt*(-np.sin(theta)*np.cos(self.OMEGA) - np.cos(theta)*np.cos(self.i)*np.sin(self.OMEGA))

        vy = vr*( np.cos(theta)*np.sin(self.OMEGA) + np.sin(theta)*np.cos(self.i)*np.cos(self.OMEGA)) +\
             vt*( np.cos(theta)*np.cos(self.i)*np.cos(self.OMEGA) - np.sin(theta)*np.sin(self.OMEGA))

        vz = vr*np.sin(theta)*np.sin(self.i) + vt*np.cos(theta)*np.sin(self.i)

        vel_vec_bi = np.array([vx, vy, vz])
        return vel_vec_bi


class Test_Approach_Neptune:

    approach = Approach("NEPTUNE",
                        v_inf_vec_icrf=np.array([17.78952518,  8.62038536,  3.15801163]),
                        rp=(24764+400)*1e3, psi=np.pi/2, h_EI=1000e3)

    def test_R1_0(self):

        assert (self.approach.R1(0) == np.array([[1,0,0],
                                               [0, np.cos(0), np.sin(0)],
                                               [0, -np.sin(0), np.cos(0)]])).all()
    def test_R1_90(self):

        assert (self.approach.R1(np.pi/2) == np.array([[1, 0, 0],
                                                 [0, np.cos(np.pi/2), np.sin(np.pi/2)],
                                                 [0, -np.sin(np.pi/2), np.cos(np.pi/2)]])).all()

    def test_R2_90(self):
        assert (self.approach.R2(np.pi/2) == np.array([[np.cos(np.pi/2), 0,  -np.sin(np.pi/2)],
                                                         [0, 1, 0],
                                                         [np.sin(np.pi/2), 0,  np.cos(np.pi/2)]])).all()

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

    def test_rp_vec_bi(self):

        ans1 =  (np.matmul(self.approach.R2(self.approach.phi_2),
                 np.matmul(self.approach.R3(self.approach.phi_1), self.approach.rp_vec_bi.T))).T
        ans2 = self.approach.rp_vec_bi_dprime

        assert abs(ans1 - ans2).all() < 1e-6

    def test_rp_vec_bi_magnitude(self):
       assert abs(LA.linalg.norm(self.approach.rp_vec_bi) - self.approach.rp) < 1e-8

    def test_e_vec_bi(self):
       ans1 = self.approach.e_vec_bi
       ans2 = np.array([1.15338569, -2.18462991, 0.1522179])
       assert max(abs(ans1-ans2)) < 1e-6

    def test_h(self):
        assert abs(self.approach.h - 773198144434.8495)/773198144434.8495 < 1e-8

    def test_h_vec_bi_unit(self):
        ans1 = self.approach.h_vec_bi_unit
        ans2 = np.array([-0.15179949, -0.01127846, 0.98834696])
        assert max(abs(ans1-ans2)) < 1e-6

    def test_i(self):
        assert abs(self.approach.i*180/np.pi - 8.75547878) < 1e-4

    def test_N_vec_bi_unit(self):
        ans1 = self.approach.N_vec_bi_unit
        ans2 = np.array([ 0.01127846, -0.15179949, 0.0])
        assert max(abs(ans1 - ans2)) < 1e-6

    def test_OMEGA(self):
        assert abs(self.approach.OMEGA - 4.723667679) < 1e-6

    def test_omega(self):
        assert abs(self.approach.omega - 1.43110144) < 1e-6

    def test_T_vec_bi_unit(self):
        ans = (np.matmul(self.approach.R2(self.approach.phi_2),
                 np.matmul(self.approach.R3(self.approach.phi_1), self.approach.T_vec_bi_unit.T))).T

        assert abs(ans[0]) < 1e-2
        assert abs(ans[1] + 1) < 0.02
        assert abs(ans[2]) < 1e-2

    def test_R_vec_bi_unit(self):
        ans = (np.matmul(self.approach.R2(self.approach.phi_2),
                 np.matmul(self.approach.R3(self.approach.phi_1), self.approach.R_vec_bi_unit.T))).T

        assert abs(abs(ans[0]) - 1) < 0.02
        assert abs(ans[1])  < 1e-2
        assert abs(ans[2]) < 1e-2

    def test_pos_vec_bi(self):
        pass

    def test_theta_star_entry(self):
        delta =  self.approach.r_mag_bi(self.approach.theta_star_entry)- self.approach.r_EI
        assert delta < 1e-4

    def test_r_vec_entry_bi(self):
        delta =  abs(LA.linalg.norm(self.approach.r_vec_entry_bi) - self.approach.r_EI)
        assert delta < 1e-4

    def test_v_entry_mag(self):
        assert abs(self.approach.v_entry_inertial_mag - 30567.901209444415) < 1e-6

    def test_gamma_entry_inertial(self):
        assert abs(self.approach.gamma_entry_inertial + 0.16007123941838497) < 1e-6

    def test_v_vec_entry_bi(self):
        assert abs(LA.linalg.norm(self.approach.v_vec_entry_bi) - 30567.901209444415) < 1e-6

    def test_latitude_entry(self):
        ans = self.approach.latitude_e

    def test_longitude_entry(self):
        ans = self.approach.longitude_entry
        pass

    def test_v_entry_atm_bi(self):
        ans1 = self.approach.v_entry_atm_bi
        ans2 = self.approach.v_entry_atm_bi_mag/1e3
        ans3 = self.approach.v_entry_inertial_mag
        ans4 = self.approach.i*180/np.pi
        pass



