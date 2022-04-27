# SOURCE FILENAME : arrival.py
# AUTHOR          : Athul Pradeepkumar Girija, apradee@purdue.edu
# DATE MODIFIED   : 04/19/2022, 07:54 MT
# REMARKS         : Compute the arrival v_inf vector in ICRF given two encounter dates and planets
#                   (last planetary encounter prior to arrival, arrival date)

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
import numpy as np

from poliastro.bodies import Sun
from poliastro import iod
from astropy import units as u
from numpy import linalg as LA

from planet import Planet

solar_system_ephemeris.set('jpl')

class Arrival:
	"""

	Compute the arrival declination from either a user-supplied v_inf vector in ICRF,
	or a v_inf_vector computed from a Lambert arc (last planetary encounter prior to
	arrival anf arrival date)

	"""

	def init__(self):
		self.v_inf_vec = None
		self.v_inf_mag = None
		self.v_inf_vec_unit = None
		self.north_pole = None
		self.angle = None
		self.declination = None


	def set_vinf_vec_manually(self, arrivalPlanet, v_inf_vec_ICRF_kms):
		"""
		Set arrival v_inf_vec manually if available
		"""

		self.v_inf_vec = v_inf_vec_ICRF_kms
		self.v_inf_mag = LA.norm(self.v_inf_vec)

		# compute arrival vinf unit vector
		self.v_inf_vec_unit = self.v_inf_vec / self.v_inf_mag

		# compute arrival planet north pole vector
		planetObj = Planet(arrivalPlanet)
		a0 = planetObj.a0
		d0 = planetObj.d0
		self.north_pole = np.array([np.cos(d0) * np.cos(a0), np.cos(d0) * np.sin(a0), np.sin(d0)])

		# compute angle between vinf vector and north pole vector
		self.angle = np.arccos(np.dot(self.north_pole, self.v_inf_vec_unit)) * 180 / np.pi

		# compute declination
		self.declination = 90 - self.angle

	def set_vinf_vec_from_lambert_arc(self, lastFlybyPlanet, arrivalPlanet, lastFlybyDate, arrivalDate,
                                    M=0, numiter=100, rtol=1e-6):

		self.lastFlybyPlanet_eph = get_body_barycentric_posvel(lastFlybyPlanet, lastFlybyDate)
		self.arrivalPlanet_eph = get_body_barycentric_posvel(arrivalPlanet, arrivalDate)

		self.TOF = arrivalDate - lastFlybyDate

		self.lastFlybyPlanet_pos = np.array([self.lastFlybyPlanet_eph[0].x.value,
		                                     self.lastFlybyPlanet_eph[0].y.value,
		                                     self.lastFlybyPlanet_eph[0].z.value])

		self.lastFlybyPlanet_vel = np.array([self.lastFlybyPlanet_eph[1].x.value / 86400,
		                                     self.lastFlybyPlanet_eph[1].y.value / 86400,
		                                     self.lastFlybyPlanet_eph[1].z.value / 86400])

		self.arrivalPlanet_pos = np.array([self.arrivalPlanet_eph[0].x.value,
		                                     self.arrivalPlanet_eph[0].y.value,
		                                     self.arrivalPlanet_eph[0].z.value])

		self.arrivalPlanet_vel = np.array([self.arrivalPlanet_eph[1].x.value / 86400,
		                                     self.arrivalPlanet_eph[1].y.value / 86400,
		                                     self.arrivalPlanet_eph[1].z.value / 86400])

		(self.v_dep, self.v_arr), = iod.izzo.lambert(Sun.k, self.lastFlybyPlanet_pos * u.km,
		                                     self.arrivalPlanet_pos * u.km,
		                                     self.TOF,
		                                     M=M, numiter=numiter, rtol=rtol)

		self.v_inf_vec = self.v_arr.value - self.arrivalPlanet_vel
		self.v_inf_mag = LA.norm(self.v_inf_vec)

		# compute arrival vinf unit vector
		self.v_inf_vec_unit = self.v_inf_vec/self.v_inf_mag

		# compute arrival planet north pole vector
		planetObj = Planet(arrivalPlanet)
		a0 = planetObj.a0
		d0 = planetObj.d0
		self.north_pole = np.array([np.cos(d0)*np.cos(a0), np.cos(d0)*np.sin(a0), np.sin(d0)])

		# compute angle between vinf vector and north pole vector
		self.angle = np.arccos(np.dot(self.north_pole, self.v_inf_vec_unit)) * 180 / np.pi

		self.declination = 90 - self.angle

	def compute_v_inf_vector(self):
		return self.v_inf_vec

	def compute_v_inf_mag(self):
		return self.v_inf_mag

	def compute_declination(self):
		return self.declination


class Test_Arrival_specified_vinf_vec:

	arrival = Arrival()
	arrival.set_vinf_vec_manually("NEPTUNE", np.array([17.78952518,  8.62038536,  3.15801163]))

	def test_compute_v_inf_mag(self):
		v_inf_mag = self.arrival.compute_v_inf_mag()
		assert abs(v_inf_mag - 20.01877337298844) < 1e-6

	def test_compute_declination(self):
		declination = self.arrival.compute_declination()
		assert (abs(declination) - 8.755478798) < 1e-6


class Test_Arrival_Neptune_2039:

	arrival = Arrival()
	arrival.set_vinf_vec_from_lambert_arc('JUPITER',
	                                  'NEPTUNE',
	                                  Time("2032-06-29 00:00:00", scale='tdb'),
	                                  Time("2039-01-03 00:00:00", scale='tdb'))

	def test_compute_v_inf_vector(self):
		v_inf_vector = self.arrival.compute_v_inf_vector()
		assert abs(v_inf_vector[0] - 17.78952518) < 1e-6
		assert abs(v_inf_vector[1] - 8.62038536) < 1e-6
		assert abs(v_inf_vector[2] - 3.15801163) < 1e-6

	def test_compute_v_inf_mag(self):
		v_inf_mag = self.arrival.compute_v_inf_mag()
		assert abs(v_inf_mag - 20.01877337298844) < 1e-6

	def test_compute_declination(self):
		declination = self.arrival.compute_declination()
		assert (abs(declination) - 8.755478798) < 1e-6


class Test_Arrival_Uranus_2043:

	arrival = Arrival()
	arrival.set_vinf_vec_from_lambert_arc('JUPITER',
	                                  'URANUS',
	                                  Time("2036-03-28 00:00:00", scale='tdb'),
	                                  Time("2043-05-17 00:00:00", scale='tdb'))

	def test_compute_v_inf_mag(self):
		v_inf_mag = self.arrival.compute_v_inf_mag()
		assert abs(v_inf_mag - 8.41) < 1e-2

	def test_compute_declination(self):
		declination = self.arrival.compute_declination()
		assert (abs(declination) - 48) < 1.0


class Test_Arrival_Neptune_2043:

	arrival = Arrival()
	arrival.set_vinf_vec_from_lambert_arc('JUPITER',
	                                  'NEPTUNE',
	                                  Time("2033-08-22 00:00:00", scale='tdb'),
	                                  Time("2043-04-28 00:00:00", scale='tdb'))

	def test_compute_v_inf_mag(self):
		v_inf_mag = self.arrival.compute_v_inf_mag()
		assert abs(v_inf_mag - 11.4) < 1e-1

	def test_compute_declination(self):
		declination = self.arrival.compute_declination()
		assert (abs(declination) - 9.1) < 0.5


class Test_Arrival_Saturn_2034:

	arrival = Arrival()
	arrival.set_vinf_vec_from_lambert_arc('EARTH',
	                                  'SATURN',
	                                  Time("2031-09-03 00:00:00", scale='tdb'),
	                                  Time("2034-12-30 00:00:00", scale='tdb'))

	def test_compute_v_inf_mag(self):
		v_inf_mag = self.arrival.compute_v_inf_mag()
		pass


class Test_Arrival_Uranus_2045:

	arrival = Arrival()
	arrival.set_vinf_vec_from_lambert_arc('JUPITER',
	                                  'URANUS',
	                                  Time("2036-04-29 00:00:00", scale='tdb'),
	                                  Time("2045-05-03 00:00:00", scale='tdb'))

	def test_compute_v_inf_mag(self):
		v_inf_mag = self.arrival.compute_v_inf_mag()
		assert abs(v_inf_mag - 5.96) < 1e-1

	def test_compute_declination(self):
		declination = self.arrival.compute_declination()
		assert (declination + 49.69) < 0.5