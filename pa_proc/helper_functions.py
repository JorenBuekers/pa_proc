import pandas as pd
import numpy as np
import logging
import sys
import exit

def calculate_vector_magnitude(data, minus_one = False, round_negative_to_zero = False, dtype = np.float32):
	
    # Code obtained from: https://github.com/shaheen-syed/ActiGraph-ActiWave-Analysis/

	"""
	Calculate vector magnitude of acceleration data
	the vector magnitude of acceleration is calculated as the Euclidian Norm

	sqrt(y^2 + x^2 + z^2)

	if minus_one is set to True then it it is the Euclidian Norm Minus One

	sqrt(y^2 + x^2 + z^2) - 1

	Parameters
	----------
	data : numpy array (acceleration values, axes)
		numpy array with acceleration data
	minus_one : Boolean (optional)
		If set to True, the calculate the vector magnitude minus one, also known as the ENMO (Euclidian Norm Minus One)
	round_negative_to_zero : Boolean (optional)
		If set to True, round negative values to zero
	dtype = mumpy data type (optional)
		set the data type of the return array. Standard float 16, but can be set to better precision
	
	Returns
	-------
	vector_magnitude : numpy array (acceleration values, 1)(np.float)
		numpy array with vector magnitude of the acceleration
	"""

	# change dtype of array to float32 (also to hold scaled data correctly). The original unscaled data is stored as int16, but when we want to calculate the vector we exceed the values that can be stored in 16 bit
	data = data.astype(dtype = np.float32)

	try:

		# calculate the vector magnitude on the whole array
		vector_magnitude = np.sqrt(np.sum(np.square(data), axis=1)).astype(dtype=dtype)

		# check if minus_one is set to True, if so, we need to calculate the ENMO
		if minus_one:
			vector_magnitude -= 1

		# if set to True, round negative values to zero
		if round_negative_to_zero:
			vector_magnitude = vector_magnitude.clip(lower=0)

		# reshape the array into number of acceleration values, 1 column
		return vector_magnitude.reshape(data.shape[0], 1)
		

	except Exception as e:
		
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)