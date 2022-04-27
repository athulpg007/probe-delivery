# SOURCE FILENAME : deflection.py
# AUTHOR          : Athul Pradeepkumar Girija, apradee@purdue.edu
# DATE MODIFIED   : 04/23/2022, 18:30 MT
# REMARKS         : Compute the deflection maneuver delta-v
#                   following Kyle's dissertation Chapter 2

import numpy as np
from numpy import linalg as LA

from planet import Planet
from approach import Approach

class Deflection:

    def __init__(self, arrivalPlanet, v_inf_vec_icrf,
                                      rp_probe, psi_probe,  h_EI_probe,
                                      rp_space, psi_space,
                                      r_dv_rp):

        self.planetObj = Planet(arrivalPlanet)
        self.probe = Approach(arrivalPlanet, v_inf_vec_icrf, rp_probe, psi_probe, is_entrySystem=True, h_EI=h_EI_probe)
        self.space = Approach(arrivalPlanet, v_inf_vec_icrf, rp_space, psi_space)
        self.r_dv_rp = r_dv_rp
        self.r_dv = self.r_dv_rp * self.planetObj.RP

        self.theta_star_dv_probe = -1*np.arccos(((self.probe.h**2/(self.planetObj.GM * self.r_dv)) - 1)*\
                                                (1.0/self.probe.e))

        self.r_vec_dv = self.probe.pos_vec_bi(self.theta_star_dv_probe)
        self.r_vec_dv_unit = self.r_vec_dv / LA.linalg.norm(self.r_vec_dv)

        self.delta_theta_star_probe = abs(self.theta_star_dv_probe)
        self.delta_theta_star_space = np.arccos(np.dot(self.space.rp_vec_bi_unit, self.r_vec_dv_unit))

        self.P_probe = self.probe.a*(1 - self.probe.e**2)

        self.f_probe = 1 - (self.probe.rp/self.P_probe)*(1 - np.cos(self.delta_theta_star_probe))
        self.g_probe = ((self.r_dv*self.probe.rp)/(np.sqrt(self.planetObj.GM*self.P_probe)))*np.sin(self.delta_theta_star_probe)

        self.v_vec_dv_probe = (self.probe.rp_vec_bi - self.f_probe*self.r_vec_dv)/self.g_probe

        self.C = np.sqrt(self.space.rp**2 + self.r_dv**2 - 2*self.space.rp*self.r_dv*np.cos(self.delta_theta_star_space))
        self.S = 0.5*(self.space.rp + self.r_dv + self.C)
        self.alpha_prime = 2*np.arcsinh(np.sqrt(self.S/(2*abs(self.space.a))))
        self.beta_prime  = 2*np.arcsinh(np.sqrt((self.S-self.C)/(2*abs(self.space.a))))

        self.P1 = (4*abs(self.space.a)*(self.S - self.space.rp)*(self.S - self.r_dv))/(self.C**2)
        self.P2 = np.sinh(0.5*(self.alpha_prime + self.beta_prime))

        self.P_space = self.P1*self.P2**2

        self.f_space = 1 - (self.space.rp / self.P_space) * (1 - np.cos(self.delta_theta_star_space))
        self.g_space = ((self.r_dv * self.space.rp) / (np.sqrt(self.planetObj.GM * self.P_space))) * np.sin(self.delta_theta_star_space)

        self.v_vec_dv_space = (self.space.rp_vec_bi - self.f_space * self.r_vec_dv) / self.g_space

        self.v_vec_dv_maneuver = self.v_vec_dv_space - self.v_vec_dv_probe

        self.H1 = np.sqrt((self.probe.e-1)/(self.probe.e+1))*np.tan(0.5*self.delta_theta_star_probe)
        self.H_dv_probe = 2*np.arctanh(self.H1)

        self.T1 = np.sqrt(abs(self.probe.a)**3/self.planetObj.GM)
        self.T2 = abs(self.probe.e*np.sinh(self.H_dv_probe) - self.H_dv_probe)

        self.TOF_probe = self.T1*self.T2

        self.U1 = np.sqrt(abs(self.space.a)**3/self.planetObj.GM)
        self.U2 = (np.sinh(self.alpha_prime) - self.alpha_prime) - (np.sinh(self.beta_prime) - self.beta_prime)

        self.TOF_space = self.U1*self.U2

class Test_Deflection_Neptune:

    deflection = Deflection("NEPTUNE", v_inf_vec_icrf=np.array([17.78952518,  8.62038536,  3.15801163]),
                                       rp_probe=(24622+400)*1e3,  psi_probe=3*np.pi/2, h_EI_probe=1000e3,
                                       rp_space=(24622+4000)*1e3, psi_space=np.pi/2,
                                       r_dv_rp=1000)

    def test_theta_star_dv_probe(self):
        ans = self.deflection.theta_star_dv_probe
        rdv = self.deflection.probe.r_mag_bi(ans)/(self.deflection.planetObj.RP)
        assert abs(rdv - self.deflection.r_dv_rp) < 1e-6

    def test_delta_theta_star_probe(self):
        ans = self.deflection.delta_theta_star_probe
        pass

    def test_delta_theta_star_space(self):
        ans = self.deflection.delta_theta_star_space
        pass

    def test_v_vec_dv_probe(self):
        ans = self.deflection.v_vec_dv_probe
        pass

    def test_v_vec_dv_space(self):
        ans = self.deflection.v_vec_dv_space
        pass

    def test_v_vec_dv_maneuver(self):
        ans1 = self.deflection.v_vec_dv_maneuver
        ans2 = LA.linalg.norm(ans1)
        pass

    def test_ad_hoc(self):
        DV = self.deflection.v_vec_dv_maneuver
        dv = LA.linalg.norm(DV)
        TOF_probe = self.deflection.TOF_probe / 86400
        TOF_space = self.deflection.TOF_space / 86400
        pass