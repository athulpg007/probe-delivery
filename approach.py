# SOURCE FILENAME : approach.py
# AUTHOR          : Athul Pradeepkumar Girija, apradee@purdue.edu
# DATE MODIFIED   : 04/26/2022, 21:23 MT
# REMARKS         : Compute the probe arrival trajectory
#                   following Kyle's dissertation Chapter 2

import numpy as np
from numpy import linalg as LA
from planet import Planet


class Approach:

	def __init__(self, arrivalPlanet, v_inf_vec_icrf_kms, rp, psi,
				       is_entrySystem=False, h_EI=None):
		"""
		Compute the probe/spacecraft approach trajectory for a given
		v_inf_vec, periapsis radius, psi.
		"""

		self.planetObj = Planet(arrivalPlanet)
		self.a0 = self.planetObj.a0
		self.d0 = self.planetObj.d0

		self.v_inf_vec_icrf_kms = v_inf_vec_icrf_kms
		self.rp = rp
		self.psi = psi

		self.v_inf_mag_kms = LA.norm(self.v_inf_vec_icrf_kms)
		self.v_inf_mag = self.v_inf_mag_kms*1e3

		self.a = -self.planetObj.GM/(self.v_inf_mag**2)
		self.e = 1 - self.rp/self.a
		self.beta = np.arccos(1.0/self.e)

		self.v_inf_vec_bi_kms = self.ICRF_to_BI(self.v_inf_vec_icrf_kms)
		self.v_inf_vec_bi = self.v_inf_vec_bi_kms * 1e3
		self.v_inf_vec_bi_mag_kms = LA.norm(self.v_inf_vec_bi_kms)

		self.v_inf_vec_bi_unit = self.v_inf_vec_bi_kms / LA.norm(self.v_inf_vec_bi_kms)

		self.phi_1 = np.arctan(self.v_inf_vec_bi[1] / self.v_inf_vec_bi[0])
		self.v_inf_vec_bi_prime = (np.matmul(self.R3(self.phi_1), self.v_inf_vec_bi.T)).T

		self.phi_2 = np.arctan(self.v_inf_vec_bi_prime[0] / self.v_inf_vec_bi_prime[2])
		self.phi_2_analytic = np.arctan((self.v_inf_vec_bi[0] * np.cos(self.phi_1) + \
									     self.v_inf_vec_bi[1] * np.sin(self.phi_1)) / \
								         self.v_inf_vec_bi[2])

		self.rp_vec_bi_x_unit = np.cos(self.phi_1)*(np.sin(self.beta) * np.cos(self.psi) * np.cos(self.phi_2) + np.cos(self.beta) * np.sin(self.phi_2)) - \
								np.sin(self.phi_1) * np.sin(self.beta) * np.sin(self.psi)
		self.rp_vec_bi_y_unit = np.sin(self.phi_1)*(np.sin(self.beta) * np.cos(self.psi) * np.cos(self.phi_2) + np.cos(self.beta) * np.sin(self.phi_2)) + \
								np.cos(self.phi_1) * np.sin(self.beta) * np.sin(self.psi)
		self.rp_vec_bi_z_unit = np.cos(self.beta) * np.cos(self.phi_2) - np.sin(self.beta) * np.cos(self.psi) * np.sin(self.phi_2)

		self.rp_vec_bi_unit = np.array([self.rp_vec_bi_x_unit,
										self.rp_vec_bi_y_unit,
										self.rp_vec_bi_z_unit])

		self.rp_vec_bi = self.rp * self.rp_vec_bi_unit

		self.rp_vec_bi_dprime = self.BI_to_BI_dprime(self.rp_vec_bi)
		self.rp_vec_bi_dprime_analytic = self.rp * np.array([np.sin(self.beta) * np.cos(self.psi),
													np.sin(self.beta) * np.sin(self.psi),
													np.cos(self.beta)])

		self.e_vec_bi = self.e * self.rp_vec_bi_unit
		self.e_vec_bi_unit = self.e_vec_bi / LA.linalg.norm(self.e_vec_bi)

		self.h = np.sqrt(self.a * self.planetObj.GM * (1 - self.e**2))

		self.h_vec_bi = np.cross(self.rp_vec_bi, self.v_inf_vec_bi)
		self.h_vec_bi_unit = self.h_vec_bi / LA.norm(self.h_vec_bi)

		self.i = np.arccos(self.h_vec_bi_unit[2])

		self.N_vec_bi = np.cross(np.array([0, 0, 1]), self.h_vec_bi)
		self.N_vec_bi_unit = self.N_vec_bi / LA.norm(self.N_vec_bi)

		if self.N_vec_bi_unit[1] >= 0:
			self.OMEGA = np.arccos(self.N_vec_bi_unit[0])
		else:
			self.OMEGA = 2*np.pi - np.arccos(self.N_vec_bi_unit[0])

		if self.e_vec_bi_unit[2] >= 0:
			self.omega = np.arccos(np.dot(self.N_vec_bi_unit, self.e_vec_bi_unit))
		else:
			self.omega = 2*np.pi - np.arccos(np.dot(self.N_vec_bi_unit, self.e_vec_bi_unit))

		self.N_ref_bi = np.array([0, 0, 1])
		self.S_vec_bi = self.v_inf_vec_bi
		self.S_vec_bi_unit = self.v_inf_vec_bi_unit

		self.T_vec_bi = np.cross(self.S_vec_bi, self.N_ref_bi)
		self.T_vec_bi_unit = self.T_vec_bi / LA.norm(self.T_vec_bi)
		self.T_vec_bi_unit_dprime = self.BI_to_BI_dprime(self.T_vec_bi_unit)

		self.R_vec_bi = np.cross(self.S_vec_bi, self.T_vec_bi)
		self.R_vec_bi_unit = self.R_vec_bi / LA.norm(self.R_vec_bi)
		self.R_vec_bi_unit_dprime = self.BI_to_BI_dprime(self.R_vec_bi_unit)

		self.b_mag = abs(self.a) * np.sqrt(self.e ** 2 - 1)
		self.b_plane_angle_theta = self.psi + np.pi / 2

		self.B_vec_bi = np.cross(self.S_vec_bi, self.h_vec_bi)
		self.B_vec_bi_unit = self.B_vec_bi / LA.norm(self.B_vec_bi)


		if is_entrySystem == True:
			self.h_EI = h_EI
			self.r_EI = self.planetObj.RP + self.h_EI

			self.theta_star_entry = -1 * np.arccos(((self.h ** 2 / (self.planetObj.GM * self.r_EI)) - 1) * (1.0 / self.e))

			self.r_vec_entry_bi = self.pos_vec_bi(self.theta_star_entry)
			self.r_vec_entry_bi_unit = self.r_vec_entry_bi / LA.linalg.norm(self.r_vec_entry_bi)
			self.r_vec_entry_bi_mag = LA.linalg.norm(self.r_vec_entry_bi)

			self.v_entry_inertial_mag = np.sqrt(self.v_inf_mag**2 + 2 * self.planetObj.GM / self.r_EI)

			self.gamma_entry_inertial = -1 * np.arccos(self.h / (self.r_EI * self.v_entry_inertial_mag))

			self.v_vec_entry_bi = self.vel_vec_bi(self.theta_star_entry)
			self.v_vec_entry_bi_unit = self.v_vec_entry_bi / LA.linalg.norm(self.v_vec_entry_bi)

			self.gamma_entry_inertial_check = np.pi / 2 - \
											  np.arccos(np.dot(self.r_vec_entry_bi_unit, self.v_vec_entry_bi_unit))

			self.latitude_entry_bi = np.arcsin(self.r_vec_entry_bi_unit[2])
			self.longitude_entry_bi = np.arctan(self.r_vec_entry_bi_unit[1] / self.r_vec_entry_bi_unit[0])

			self.xi_1 = np.arctan(self.r_vec_entry_bi_unit[1]/self.r_vec_entry_bi_unit[0])
			self.latitude_entry_bi_analytic = np.arctan(self.r_vec_entry_bi_unit[2]/
														(self.r_vec_entry_bi_unit[0]*np.cos(self.xi_1) + self.r_vec_entry_bi_unit[1]*np.sin(self.xi_1)))

			self.v_vec_entry_atm = self.vel_vec_entry_atm()
			self.v_entry_atm_mag = LA.linalg.norm(self.v_vec_entry_atm)
			self.v_vec_entry_atm_unit = self.v_vec_entry_atm / self.v_entry_atm_mag

			self.gamma_entry_atm = np.pi / 2 - \
								   np.arccos(np.dot(self.r_vec_entry_bi_unit, self.v_vec_entry_atm_unit))


		else:
			self.theta_star_periapsis = -1 * np.arccos(((self.h ** 2 / (self.planetObj.GM * self.rp)) - 1) * (1.0 / self.e))

	def R1(self, theta):
		return np.array([[1, 0, 0],
						 [0, np.cos(theta), np.sin(theta)],
						 [0, -np.sin(theta), np.cos(theta)]])

	def R2(self, theta):
		return np.array([[np.cos(theta), 0, -np.sin(theta)],
						 [0, 1, 0],
						 [np.sin(theta), 0, np.cos(theta)]])

	def R3(self, theta):
		return np.array([[np.cos(theta), np.sin(theta), 0],
						 [-np.sin(theta), np.cos(theta), 0],
						 [0, 0, 1]])

	def ICRF_to_BI(self, X_ICRF):
		R1R3 = np.matmul(self.R1(np.pi/2 - self.d0), self.R3(np.pi / 2 + self.a0))
		return np.matmul(R1R3, X_ICRF.T).T

	def BI_to_BI_dprime(self, X_ICRF):
		R2R3 = np.matmul(self.R2(self.phi_2), self.R3(self.phi_1))
		return np.matmul(R2R3, X_ICRF.T).T

	def pos_vec_bi(self, theta_star):

		r = (self.h**2 / self.planetObj.GM) / (1 + self.e*np.cos(theta_star))
		theta = theta_star + self.omega

		rx_unit = np.cos(self.OMEGA)*np.cos(theta) - np.sin(self.OMEGA)*np.cos(self.i)*np.sin(theta)
		ry_unit = np.sin(self.OMEGA)*np.cos(theta) + np.cos(self.OMEGA)*np.cos(self.i)*np.sin(theta)
		rz_unit = np.sin(self.i)*np.sin(theta)

		pos_vec_bi = r*np.array([rx_unit, ry_unit, rz_unit])
		return pos_vec_bi


	def pos_vec_bi_dprime(self, theta_star):

		pos_vec_bi = self.pos_vec_bi(theta_star)

		pos_vec_bi_dprime = (np.matmul(self.R2(self.phi_2),
		        			 np.matmul(self.R3(self.phi_1), pos_vec_bi))).T

		return pos_vec_bi_dprime




	def r_mag_bi(self, theta_star):
		r_mag_bi = (self.h**2 / self.planetObj.GM) / (1 + self.e*np.cos(theta_star))
		return r_mag_bi

	def vel_vec_bi(self, theta_star):

		r_mag_bi = self.r_mag_bi(theta_star)
		v_mag_bi = np.sqrt((self.v_inf_mag)**2 + 2*self.planetObj.GM/r_mag_bi)
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

	def vel_vec_entry_atm(self):
		v_entry_atm_x = self.v_vec_entry_bi[0] - \
		          self.r_vec_entry_bi_mag * self.planetObj.OMEGA * \
		          np.cos(self.latitude_entry_bi) * (-np.sin(self.longitude_entry_bi))
		v_entry_atm_y = self.v_vec_entry_bi[1] - \
		          self.r_vec_entry_bi_mag * self.planetObj.OMEGA * \
		          np.cos(self.latitude_entry_bi) * (np.cos(self.longitude_entry_bi))
		v_entry_atm_z = self.v_vec_entry_bi[2]

		vel_vec_entry_atm = np.array([v_entry_atm_x, v_entry_atm_y, v_entry_atm_z])
		return vel_vec_entry_atm



class Test_Approach:

	approach = Approach("NEPTUNE",
						v_inf_vec_icrf_kms=np.array([17.78952518, 8.62038536, 3.15801163]),
						rp=(24622+400)*1e3, psi=3*np.pi/2,
						is_entrySystem=True, h_EI=1000e3)

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


	def test_v_inf_mag(self):
		assert abs(self.approach.v_inf_mag_kms - 20.018) < 1e-3
		assert abs(self.approach.v_inf_mag - 20018) < 1

	def test_a(self):
		assert ((abs(self.approach.a) - abs(-17059283.6903)) / abs(-17059283.6903)) < 1e-6

	def test_e(self):
		assert (abs(self.approach.e - 2.466767) / 2.466767) < 1e-4

	def test_beta(self):
		assert (abs(self.approach.beta - 1.153392) / 1.153392) < 1e-4

	def test_v_inf_bi(self):
		assert abs(self.approach.v_inf_vec_bi_mag_kms - 20.018) < 1e-3
		assert abs(np.arcsin(self.approach.v_inf_vec_bi_unit[2])*180/np.pi - 8.7554787) < 1e-4

	def test_phi(self):
		assert abs(self.approach.phi_1 - 0.07416) < 1e-3
		assert abs(self.approach.phi_2 - self.approach.phi_2_analytic) < 1e-8

	def test_rp_vec(self):
		assert (self.approach.rp_vec_bi_dprime[0] - self.approach.rp_vec_bi_dprime_analytic[0]) < 1e-6
		assert (self.approach.rp_vec_bi_dprime[1] - self.approach.rp_vec_bi_dprime_analytic[1]) < 1e-6
		assert (self.approach.rp_vec_bi_dprime[2] - self.approach.rp_vec_bi_dprime_analytic[2]) < 1e-6

	def test_e_vec_bi_unit(self):
		assert (self.approach.e_vec_bi_unit[0] - self.approach.rp_vec_bi_unit[0]) < 1e-6
		assert (self.approach.e_vec_bi_unit[1] - self.approach.rp_vec_bi_unit[1]) < 1e-6
		assert (self.approach.e_vec_bi_unit[2] - self.approach.rp_vec_bi_unit[2]) < 1e-6

	def test_h_vec_bi_unit(self):
		assert abs(np.arccos(self.approach.h_vec_bi_unit[2])*180/np.pi - 8.7554787) < 1e-4

	def test_N_vec_bi_unit(self):
		assert abs(self.approach.N_vec_bi_unit[0] - 0.07409418) < 1e-6
		assert abs(self.approach.N_vec_bi_unit[1] - -0.99725125) < 1e-6
		assert abs(self.approach.N_vec_bi_unit[2] - 0.) < 1e-6

	def test_i(self):
		assert abs(self.approach.i*180/np.pi - 8.75547878) < 1e-4

	def test_OMEGA(self):
		assert abs(self.approach.OMEGA - 4.78655112) < 1e-6

	def test_omega(self):
		assert abs(self.approach.omega - 0.41740417) < 1e-6

	def test_B_plane(self):
		assert abs(LA.norm(self.approach.R_vec_bi_unit) - 1) < 1e-8
		assert abs(LA.norm(self.approach.S_vec_bi_unit) - 1) < 1e-8
		assert abs(LA.norm(self.approach.T_vec_bi_unit) - 1) < 1e-8
		assert abs(self.approach.T_vec_bi_unit_dprime[1] + 1)  < 1e-8
		assert abs(self.approach.R_vec_bi_unit_dprime[0] - 1)  < 1e-8
		assert np.dot(self.approach.R_vec_bi_unit, self.approach.S_vec_bi_unit) < 1e-8
		assert np.dot(self.approach.R_vec_bi_unit, self.approach.T_vec_bi_unit) < 1e-8
		assert np.dot(self.approach.S_vec_bi_unit, self.approach.T_vec_bi_unit) < 1e-8
		assert np.dot(self.approach.S_vec_bi_unit, self.approach.B_vec_bi_unit) < 1e-8

	def test_theta_star_entry(self):
		assert abs(self.approach.theta_star_entry - -0.25726498) < 1e-6

	def test_r_vec_entry_bi(self):
		delta =  abs(LA.norm(self.approach.r_vec_entry_bi) - self.approach.r_EI)
		assert delta < 1e-4

	def test_v_entry_mag(self):
		assert abs(self.approach.v_entry_inertial_mag - 30567.901209444415) < 1e-6

	def test_gamma_entry_inertial(self):
		assert abs(self.approach.gamma_entry_inertial + 0.18330372) < 1e-6

	def test_v_vec_entry_bi(self):
		assert abs(LA.linalg.norm(self.approach.v_vec_entry_bi) - 30567.901209444415) < 1e-6

	def test_latitude_entry(self):
		ans1 = self.approach.latitude_entry_bi
		ans2 = self.approach.latitude_entry_bi_analytic
		assert abs(ans1 - ans2) < 1e-8

	def test_longitude_entry(self):
		assert abs(self.approach.longitude_entry_bi + 1.33832990) < 1e-6

	def test_v_vec_entry_atm(self):
		assert abs(self.approach.v_vec_entry_atm[0] - 24906.113142) < 1e-3
		assert abs(self.approach.v_vec_entry_atm[1] - 11733.340980) < 1e-3
		assert abs(self.approach.v_vec_entry_atm[2] - 4381.2517221) < 1e-3

	def test_v_entry_atm_mag(self):
		assert abs(self.approach.v_entry_atm_mag - 27877.9685250) < 1e-4

	def test_gamma_entry_inertial_check(self):
		ans1 = self.approach.gamma_entry_inertial
		ans2 = self.approach.gamma_entry_inertial_check
		assert abs(ans1-ans2) < 1e-6

	def test_gamma_entry_atm(self):
		assert abs(self.approach.gamma_entry_atm + 0.2012221) < 1e-4


class Test_Approach_Orbiter:

	approach = Approach("NEPTUNE",
						v_inf_vec_icrf_kms=np.array([17.78952518, 8.62038536, 3.15801163]),
						rp=(24622+4000)*1e3, psi=3*np.pi/2)

	def test_theta_star_periapsis(self):
		ans = self.approach.theta_star_periapsis
		pass