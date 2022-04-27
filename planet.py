import numpy as np

class Planet:

	def __init__(self, planetID):
		"""
		Add north pole directions (a0, d0) in ICRF from IAU Working
		Group report for all planetary bodies.

		"""

		if planetID == 'VENUS':
			self.RP     = 6051.8000E3  
			self.OMEGA  = -2.99237E-7  
			self.GM     = 3.248599E14
			self.a0 = 272.76 * np.pi/180.0
			self.d0 = 67.16 * np.pi/180.0


		elif planetID == 'EARTH':      
			self.RP     = 6371.0000E3  
			self.OMEGA  = 7.272205E-5  
			self.GM     = 3.986004E14   
			self.a0 = 0.0 * np.pi/180.0
			self.d0 = 90.0 * np.pi/180.0

		elif planetID == 'MARS':
			self.RP     = 3389.5000E3  
			self.OMEGA  = 7.088253E-5  
			self.GM     = 4.282837E13
			self.a0 = 317.68143 * np.pi/180.0
			self.d0 = 52.88650 * np.pi/180.0   
		
		elif planetID == 'JUPITER':
			self.RP     = 69911.0E3      
			self.OMEGA  = 1.758518E-04   
			self.GM     = 1.26686534E17
			self.a0 = 268.056595 * np.pi/180.0
			self.d0 = 64.495303 * np.pi/180.0  


		if planetID == "SATURN":
			self.RP = 58232.0E3
			self.OMEGA = 1.6379E-04
			self.GM = 3.7931187E16
			self.a0 = 40.589 * np.pi / 180.0
			self.d0 = 83.537 * np.pi / 180.0
			
		elif planetID == 'TITAN':
			self.RP     = 2575.0000E3   
			self.OMEGA  = 4.5451280E-6  
			self.GM     = 8.9780000E12
			self.a0 = 39.4827 * np.pi / 180.0
			self.d0 = 83.4279 * np.pi / 180.0

		if planetID == "URANUS":
			self.GM = 5.793939E15
			self.a0 = 257.311 * np.pi/180.0
			self.d0 = -15.175 * np.pi/180.0
			self.RP = 25559.0E3
			self.OMEGA = -1.01237E-4
		
			
		if planetID == "NEPTUNE":
			self.RP = 24622.000E3
			self.OMEGA = 1.083385E-4
			self.GM = 6.8365299E15
			self.a0 = 299.36 * np.pi/180.0
			self.d0 = 43.36 * np.pi/180.0
			
			
		

		


