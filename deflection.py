# SOURCE FILENAME : deflection2.py
# AUTHOR          : Athul Pradeepkumar Girija, apradee@purdue.edu
# DATE MODIFIED   : 04/29/2022, 08:27 MT
# REMARKS         : Compute the deflection maneuver delta-v
#                   following Kyle's dissertation Chapter 2

import numpy as np
from numpy import linalg as LA

from planet import Planet
from approach import Approach

class ProbeOrbiterDeflection:

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
        self.v_vec_dv_maneuver_mag = LA.norm(self.v_vec_dv_maneuver)

        self.H1 = np.sqrt((self.probe.e-1)/(self.probe.e+1))*np.tan(0.5*self.delta_theta_star_probe)
        self.H_dv_probe = 2*np.arctanh(self.H1)

        self.T1 = np.sqrt(abs(self.probe.a)**3/self.planetObj.GM)
        self.T2 = abs(self.probe.e*np.sinh(self.H_dv_probe) - self.H_dv_probe)

        self.TOF_probe = self.T1*self.T2

        self.U1 = np.sqrt(abs(self.space.a)**3/self.planetObj.GM)
        self.U2 = (np.sinh(self.alpha_prime) - self.alpha_prime) - (np.sinh(self.beta_prime) - self.beta_prime)

        self.TOF_space = self.U1*self.U2


class ProbeProbeDeflection:

    def __init__(self, arrivalPlanet, v_inf_vec_icrf,
                                      rp_probe1, psi_probe1,  h_EI_probe1,
                                      rp_probe2, psi_probe2,  h_EI_probe2,
                                      r_dv_rp):

        self.planetObj = Planet(arrivalPlanet)
        self.probe1 = Approach(arrivalPlanet, v_inf_vec_icrf, rp_probe1, psi_probe1, is_entrySystem=True, h_EI=h_EI_probe1)
        self.probe2 = Approach(arrivalPlanet, v_inf_vec_icrf, rp_probe2, psi_probe2, is_entrySystem=True, h_EI=h_EI_probe2)
        self.r_dv_rp = r_dv_rp
        self.r_dv = self.r_dv_rp * self.planetObj.RP

        self.theta_star_dv_probe1 = -1*np.arccos(((self.probe1.h**2/(self.planetObj.GM * self.r_dv)) - 1)*\
                                                (1.0/self.probe1.e))
        self.theta_star_dv_probe2 = -1 * np.arccos(((self.probe2.h ** 2 / (self.planetObj.GM * self.r_dv)) - 1) * \
                                                   (1.0 / self.probe2.e))

        self.r_vec_dv = self.probe1.pos_vec_bi(self.theta_star_dv_probe1)
        self.r_vec_dv_unit = self.r_vec_dv / LA.linalg.norm(self.r_vec_dv)

        self.delta_theta_star_probe1 = abs(self.theta_star_dv_probe1)
        self.delta_theta_star_probe2 = abs(self.theta_star_dv_probe2)


        self.P_probe1 = self.probe1.a*(1 - self.probe1.e**2)
        self.f_probe1 = 1 - (self.probe1.rp/self.P_probe1)*(1 - np.cos(self.delta_theta_star_probe1))
        self.g_probe1 = ((self.r_dv*self.probe1.rp)/(np.sqrt(self.planetObj.GM*self.P_probe1)))*np.sin(self.delta_theta_star_probe1)
        self.v_vec_dv_probe1 = (self.probe1.rp_vec_bi - self.f_probe1*self.r_vec_dv)/self.g_probe1

        self.P_probe2 = self.probe2.a*(1 - self.probe2.e**2)
        self.f_probe2 = 1 - (self.probe2.rp / self.P_probe2) * (1 - np.cos(self.delta_theta_star_probe2))
        self.g_probe2 = ((self.r_dv * self.probe2.rp) / (np.sqrt(self.planetObj.GM * self.P_probe2))) * np.sin(self.delta_theta_star_probe2)
        self.v_vec_dv_probe2 = (self.probe2.rp_vec_bi - self.f_probe2 * self.r_vec_dv) / self.g_probe2

        self.v_vec_dv_maneuver = self.v_vec_dv_probe2 - self.v_vec_dv_probe1
        self.v_vec_dv_maneuver_mag = LA.norm(self.v_vec_dv_maneuver)

        self.H1 = np.sqrt((self.probe1.e-1)/(self.probe1.e+1))*np.tan(0.5*self.delta_theta_star_probe1)
        self.H_dv_probe1 = 2*np.arctanh(self.H1)
        self.T1 = np.sqrt(abs(self.probe1.a)**3/self.planetObj.GM)
        self.T2 = abs(self.probe1.e*np.sinh(self.H_dv_probe1) - self.H_dv_probe1)
        self.TOF_probe1 = self.T1*self.T2

        self.H2 = np.sqrt((self.probe2.e - 1) / (self.probe2.e + 1)) * np.tan(0.5 * self.delta_theta_star_probe2)
        self.H_dv_probe2 = 2 * np.arctanh(self.H2)
        self.T3 = np.sqrt(abs(self.probe2.a) ** 3 / self.planetObj.GM)
        self.T4 = abs(self.probe2.e * np.sinh(self.H_dv_probe2) - self.H_dv_probe2)
        self.TOF_probe2 = self.T3 * self.T4


class OrbiterOrbiterDeflection:

    def __init__(self, arrivalPlanet, v_inf_vec_icrf,
                                      rp_space1, psi_space1,
                                      rp_space2, psi_space2,
                                      r_dv_rp):

        self.planetObj = Planet(arrivalPlanet)
        self.space1 = Approach(arrivalPlanet, v_inf_vec_icrf, rp_space1, psi_space1)
        self.space2 = Approach(arrivalPlanet, v_inf_vec_icrf, rp_space2, psi_space2)
        self.r_dv_rp = r_dv_rp
        self.r_dv = self.r_dv_rp * self.planetObj.RP

        self.theta_star_dv_space1 = -1*np.arccos(((self.space1.h**2/(self.planetObj.GM * self.r_dv)) - 1)*\
                                                (1.0/self.space1.e))

        self.r_vec_dv = self.space1.pos_vec_bi(self.theta_star_dv_space1)
        self.r_vec_dv_unit = self.r_vec_dv / LA.linalg.norm(self.r_vec_dv)

        self.delta_theta_star_space1 = np.arccos(np.dot(self.space1.rp_vec_bi_unit, self.r_vec_dv_unit))
        self.delta_theta_star_space2 = np.arccos(np.dot(self.space2.rp_vec_bi_unit, self.r_vec_dv_unit))


        self.C1 = np.sqrt(self.space1.rp**2 + self.r_dv**2 - 2*self.space1.rp*self.r_dv*np.cos(self.delta_theta_star_space1))
        self.S1 = 0.5*(self.space1.rp + self.r_dv + self.C1)
        self.alpha_prime1 = 2*np.arcsinh(np.sqrt(self.S1/(2*abs(self.space1.a))))
        self.beta_prime1  = 2*np.arcsinh(np.sqrt((self.S1-self.C1)/(2*abs(self.space1.a))))
        self.P1 = (4*abs(self.space1.a)*(self.S1 - self.space1.rp)*(self.S1 - self.r_dv))/(self.C1**2)
        self.P2 = np.sinh(0.5*(self.alpha_prime1 + self.beta_prime1))
        self.P_space1 = self.P1*self.P2**2
        self.f_space1 = 1 - (self.space1.rp / self.P_space1) * (1 - np.cos(self.delta_theta_star_space1))
        self.g_space1 = ((self.r_dv * self.space1.rp) / (np.sqrt(self.planetObj.GM * self.P_space1))) * np.sin(self.delta_theta_star_space1)
        self.v_vec_dv_space1 = (self.space1.rp_vec_bi - self.f_space1 * self.r_vec_dv) / self.g_space1

        self.C2 = np.sqrt(self.space2.rp ** 2 + self.r_dv ** 2 - 2 * self.space2.rp * self.r_dv * np.cos(self.delta_theta_star_space2))
        self.S2 = 0.5 * (self.space2.rp + self.r_dv + self.C2)
        self.alpha_prime2 = 2 * np.arcsinh(np.sqrt(self.S2 / (2 * abs(self.space2.a))))
        self.beta_prime2 = 2 * np.arcsinh(np.sqrt((self.S2 - self.C2) / (2 * abs(self.space2.a))))
        self.P3 = (4 * abs(self.space2.a) * (self.S2 - self.space2.rp) * (self.S2 - self.r_dv)) / (self.C2 ** 2)
        self.P4 = np.sinh(0.5 * (self.alpha_prime2 + self.beta_prime2))
        self.P_space2 = self.P3 * self.P4 ** 2
        self.f_space2 = 1 - (self.space2.rp / self.P_space2) * (1 - np.cos(self.delta_theta_star_space2))
        self.g_space2 = ((self.r_dv * self.space2.rp) / (np.sqrt(self.planetObj.GM * self.P_space2))) * np.sin(self.delta_theta_star_space2)
        self.v_vec_dv_space2 = (self.space2.rp_vec_bi - self.f_space2 * self.r_vec_dv) / self.g_space2

        self.v_vec_dv_maneuver = self.v_vec_dv_space2 - self.v_vec_dv_space1
        self.v_vec_dv_maneuver_mag = LA.norm(self.v_vec_dv_maneuver)

        self.U1 = np.sqrt(abs(self.space1.a)**3/self.planetObj.GM)
        self.U2 = (np.sinh(self.alpha_prime1) - self.alpha_prime1) - (np.sinh(self.beta_prime1) - self.beta_prime1)
        self.TOF_space1 = self.U1*self.U2

        self.U3 = np.sqrt(abs(self.space2.a) ** 3 / self.planetObj.GM)
        self.U4 = (np.sinh(self.alpha_prime2) - self.alpha_prime2) - (np.sinh(self.beta_prime2) - self.beta_prime2)
        self.TOF_space2 = self.U3 * self.U4



class Test_ProbeOrbiterDeflection_Neptune:

    deflection = ProbeOrbiterDeflection("NEPTUNE", v_inf_vec_icrf=np.array([17.78952518,  8.62038536,  3.15801163]),
                                       rp_probe=(24622+400)*1e3,  psi_probe=3*np.pi/2, h_EI_probe=1000e3,
                                       rp_space=(24622+4000)*1e3, psi_space=np.pi/2,
                                       r_dv_rp=1000)

    def test_theta_star_dv_probe(self):
        ans = self.deflection.theta_star_dv_probe
        rdv = self.deflection.probe.r_mag_bi(ans)/self.deflection.planetObj.RP
        assert abs(ans + 1.9866386) < 1e-6
        assert abs(rdv - self.deflection.r_dv_rp) < 1e-6

    def test_delta_theta_star_probe(self):
        assert abs(self.deflection.delta_theta_star_probe - 1.9866386) < 1e-6

    def test_delta_theta_star_space(self):
        pass

    def test_v_vec_dv_probe(self):
        pass

    def test_v_vec_dv_space(self):
        pass

    def test_v_vec_dv_maneuver(self):
        assert abs(self.deflection.v_vec_dv_maneuver_mag - 65.7302) < 1e-2
        assert abs(self.deflection.TOF_probe / 86400 - 14.17563) < 1e-2
        assert abs(self.deflection.TOF_space / 86400 - 14.17647) < 1e-2


class Test_ProbeProbeDeflection_Neptune:

    deflection = ProbeProbeDeflection( "NEPTUNE",
                                       v_inf_vec_icrf=np.array([17.78952518,  8.62038536,  3.15801163]),
                                       rp_probe1=(24622+400)*1e3,  psi_probe1=3*np.pi/2, h_EI_probe1=1000e3,
                                       rp_probe2=(24622+400)*1e3,  psi_probe2=np.pi/2,  h_EI_probe2=1000e3,
                                       r_dv_rp=4000)


    def test_v_vec_dv_maneuver(self):
        ans1 = self.deflection.v_vec_dv_maneuver_mag
        ans2 = self.deflection.TOF_probe1/86400
        ans3 = self.deflection.TOF_probe2/86400
        ans4 = self.deflection.probe1.gamma_entry_inertial*180/np.pi
        ans5 = self.deflection.probe2.gamma_entry_inertial*180/np.pi
        ans6 = self.deflection.probe1.heading_entry_atm*180/np.pi
        ans7 = self.deflection.probe2.heading_entry_atm*180/np.pi
        pass


class Test_OrbiterOrbiterDeflection_Neptune:

    deflection = OrbiterOrbiterDeflection( "NEPTUNE",
                                       v_inf_vec_icrf=np.array([17.78952518,  8.62038536,  3.15801163]),
                                       rp_space1=(24622+4000)*1e3,  psi_space1=3*np.pi/2,
                                       rp_space2=(24622+4000)*1e3,  psi_space2=np.pi/2,
                                       r_dv_rp=1000)


    def test_v_vec_dv_maneuver(self):
        ans1 = self.deflection.v_vec_dv_maneuver_mag
        ans2 = self.deflection.TOF_space1/86400
        ans3 = self.deflection.TOF_space2/86400
        pass
