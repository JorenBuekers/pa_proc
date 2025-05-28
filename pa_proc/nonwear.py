import numpy as np

def calculate_vector_magnitude(data, minus_one = False, round_negative_to_zero = False, dtype = np.float32):

	# Code adapted from: https://github.com/shaheen-syed/ActiGraph-ActiWave-Analysis

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

	# calculate the vector magnitude on the whole array
	vector_magnitude = np.sqrt(np.sum(np.square(data), axis=1)).astype(dtype=dtype)

	# check if minus_one is set to True, if so, we need to calculate the ENMO
	if minus_one:
		vector_magnitude -= 1

	# if set to True, round negative values to zero
	if round_negative_to_zero:
		vector_magnitude = vector_magnitude.clip(min=0)

	# reshape the array into number of acceleration values, 1 column
	return vector_magnitude.reshape(data.shape[0], 1)
		
def weartime_choi2011(data, time, 
					  activity_threshold = 0, min_period_len = 90, 
					  spike_tolerance = 2,  min_window_len = 30, window_spike_tolerance = 0,  
					  use_vector_magnitude = False, axis_nr = np.nan):
    
    # Code adapted from: https://github.com/shaheen-syed/ActiGraph-ActiWave-Analysis
	#  added to allow for choosing which axis should be used for calculation
    
	"""	
	Estimate non-wear time based on Choi 2011 paper:
	Med Sci Sports Exerc. 2011 Feb;43(2):357-64. doi: 10.1249/MSS.0b013e3181ed61a3.
	Validation of accelerometer wear and nonwear time classification algorithm.
	Choi L1, Liu Z, Matthews CE, Buchowski MS.
	Description from the paper:
	1-min time intervals with consecutive zero counts for at least 90-min time window (window 1), allowing a short time intervals with nonzero counts lasting up to 2 min (allowance interval) 
	if no counts are detected during both the 30 min (window 2) of upstream and downstream from that interval; any nonzero counts except the allowed short interval are considered as wearing
	Parameters
	------------
	data: np.array((n_samples, 3 axes))
		numpy array with 60s epoch data for axis1, axis2, and axis3 (respectively X,Y, and Z axis)
	time : np.array((n_samples, 1 axis))
		numpy array with timestamps for each epoch, note that 1 epoch is 60s
	activity_threshold : int (optional)
		The activity threshold is the value of the count that is considered "zero", since we are searching for a sequence of zero counts. Default threshold is 0
	min_period_len : int (optional)
		The minimum length of the consecutive zeros that can be considered valid non wear time. Default value is 90 (since we have 60s epoch data, this equals 90 mins)
	spike_tolerance : int (optional)
		Any count that is above the activity threshold is considered a spike. The tolerence defines the number of spikes that are acceptable within a sequence of zeros. The default is 2, meaning that we allow for 2 spikes in the data, i.e. aritifical movement
	min_window_len : int (optional)
		minimum length of upstream or downstream time window (referred to as window2 in the paper) for consecutive zero counts required before and after the artifactual movement interval to be considered a nonwear time interval.
	use_vector_magnitude: Boolean (optional)
		if set to true, then use the vector magniturde of X,Y, and Z axis, otherwise, use X-axis only. Default False
	axis_nr : int (optional)
		if use_vector_magnitude is false, indicate index (0, 1 or 2) of which column represents the vertical axis
	Returns
	---------
	non_wear_vector : np.array((n_samples, 1))
		numpy array with non wear time encoded as 0, and wear time encoded as 1.
	"""


	# check if data contains at least min_period_len of data
	if len(data) < min_period_len:
		print('Epoch data contains {} samples, which is less than the {} minimum required samples'.format(len(data), min_period_len))


	# create non wear vector as numpy array with ones. now we only need to add the zeros which are the non-wear time segments
	non_wear_vector = np.ones((len(data),1), dtype = np.int16)


	"""
		ADJUST THE COUNTS IF NECESSARY
	"""


	# if use vector magnitude is set to True, then calculate the vector magnitude of axis 1, 2, and 3, which are X, Y, and Z
	if use_vector_magnitude:
		# calculate vectore
		data = calculate_vector_magnitude(data, minus_one = False, round_negative_to_zero = False)
	else:
		# if not set to true, then use axis as specified by axis_nr
		data = data[:,axis_nr]

	"""
		VARIABLES USED TO KEEP TRACK OF NON WEAR PERIODS
	"""

	# indicator for resetting and starting over
	reset = False
	# indicator for stopping the non-wear period
	stopped = False
	# indicator for starting to count the non-wear period
	start = False
	# second window validation
	window_2_invalid = False
	# starting minute for the non-wear period
	strt_nw = 0
	# ending minute for the non-wear period
	end_nw = 0
	# counter for the number of minutes with intensity between 1 and 100
	cnt_non_zero = 0
	# keep track of non wear sequences
	ranges = []

	"""
		FIND NON WEAR PERIODS IN DATA
	"""

	# loop over the data
	for paxn in range(0, len(data)):

		# get the value
		paxinten = data[paxn]

		# reset counters if reset or stopped
		if reset or stopped:	
			
			strt_nw = 0
			end_nw = 0
			start = False
			reset = False
			stopped = False
			window_2_invalid = False
			cnt_non_zero = 0

		# the non-wear period starts with a zero count
		if paxinten == 0 and start == False:
			
			# assign the starting minute of non-wear
			strt_nw = paxn
			# set start boolean to true so we know that we started the period
			start = True

		# only do something when the non-wear period has started
		if start:

			# keep track of the number of minutes with intensity that is not a 'zero' count
			if paxinten > activity_threshold:
				
				# increase the spike counter
				cnt_non_zero +=1

			# when there is a non-zero count, check the upstream and downstream window for counts
			# only when the upstream and downstream window have zero counts, then it is a valid non wear sequence
			if paxinten > 0:

				# check upstream window if there are counts, note that we skip the count right after the spike, since we allow for 2 minutes of spikes
				upstream = data[paxn + spike_tolerance: paxn + min_window_len + 1]

				# check if upstream has non zero counts, if so, then the window is invalid
				if (upstream > 0).sum() > window_spike_tolerance:
					window_2_invalid = True

				# check downstream window if there are counts, again, we skip the count right before since we allow for 2 minutes of spikes
				downstream = data[paxn - min_window_len if paxn - min_window_len > 0 else 0: paxn - 1]

				# check if downstream has non zero counts, if so, then the window is invalid
				if (downstream > 0).sum() > window_spike_tolerance:
					window_2_invalid = True

				# if the second window is invalid, we need to reset the sequence for the next run
				if window_2_invalid:
					reset = True

			# reset counter if value is "zero" again
			# if paxinten == 0:
			# 	cnt_non_zero = 0

			if paxinten <= activity_threshold:
				cnt_non_zero = 0

			# the sequence ends when there are 3 consecutive spikes, or an invalid second window (upstream or downstream), or the last value of the sequence	
			if cnt_non_zero == 3 or window_2_invalid or paxn == len(data -1):
				
				# define the end of the period
				end_nw = paxn

				# check if the sequence is sufficient in length
				if len(data[strt_nw:end_nw]) < min_period_len:
					# lenght is not sufficient, so reset values in next run
					reset = True
				else:
					# length of sequence is sufficient, set stopped to True so we save the sequence start and end later on
					stopped = True

			# if stopped is True, the sequence stopped and is valid to include in the ranges
			if stopped:
				# add ranges start and end non wear time
				ranges.append([strt_nw, end_nw])


	# convert ranges into non-wear sequence vector
	for row in ranges:
		
		# set the non wear vector according to start and end
		non_wear_vector[row[0]:row[1]] = 0			

	return non_wear_vector


def weartime_vanhees2013(data, hz = 100, min_non_wear_time_window = 60, window_overlap = 15, std_mg_threshold = 3.0, std_min_num_axes = 2 , value_range_mg_threshold = 50.0, value_range_min_num_axes = 2):
	
	# Code adapted from: https://github.com/shaheen-syed/ActiGraph-ActiWave-Analysis
	
	"""
	Estimation of non-wear time periods based on Hees 2013 paper
	Estimation of Daily Energy Expenditure in Pregnant and Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer
	Vincent T. van Hees  , Frida Renström , Antony Wright, Anna Gradmark, Michael Catt, Kong Y. Chen, Marie Löf, Les Bluck, Jeremy Pomeroy, Nicholas J. Wareham, Ulf Ekelund, Søren Brage, Paul W. Franks
	Published: July 29, 2011https://doi.org/10.1371/journal.pone.0022922
	Accelerometer non-wear time was estimated on the basis of the standard deviation and the value range of each accelerometer axis, calculated for consecutive blocks of 30 minutes. 
	A block was classified as non-wear time if the standard deviation was less than 3.0 mg (1 mg = 0.00981 m·s−2) for at least two out of the three axes or if the value range, for 
	at least two out of three axes, was less than 50 mg.
	Parameters
	----------
	data: np.array(n_samples, axes)
		numpy array with acceleration data in g values. Each column represent a different axis, normally ordered YXZ
	hz: int (optional)
		sample frequency in hertz. Indicates the number of samples per 1 second. Default to 100 for 100hz. The sample frequency is necessary to 
		know how many samples there are in a specific window. So let's say we have a window of 15 minutes, then there are hz * 60 * 15 samples
	min_non_wear_time_window : int (optional)
		minimum window length in minutes to be classified as non-wear time
	window_overlap : int (optional)
		basically the sliding window that progresses over the acceleration data. Defaults to 15 minutes.
	std_mg_threshold : float (optional)
		standard deviation threshold in mg. Acceleration axes values below or equal this threshold can be considered non-wear time. Defaults to 3.0g. 
		Note that within the code we convert mg to g.
	std_min_num_axes : int (optional) 
		minimum numer of axes used to check if acceleration values are below the std_mg_threshold value. Defaults to 2 axes; meaning that at least 2 
		axes need to have values below a threshold value to be considered non wear time
	value_range_mg_threshold : float (optional)
		value range threshold value in mg. If the range of values within a window is below this threshold (meaning that there is very little change 
		in acceleration over time) then this can be considered non wear time. Default to 50 mg. Note that within the code we convert mg to g
	value_range_min_num_axes : int (optional)
		minimum numer of axes used to check if acceleration values range are below the value_range_mg_threshold value. Defaults to 2 axes; meaning that at least 2 axes need to have a value range below a threshold value to be considered non wear time
	Returns
	---------
	non_wear_vector : np.array((n_samples, 1))
		numpy array with non wear time encoded as 0, and wear time encoded as 1.
	"""

	# number of data samples in 1 minute
	num_samples_per_min = hz * 60

	# define the correct number of samples for the window and window overlap
	min_non_wear_time_window *= num_samples_per_min
	window_overlap *= num_samples_per_min

	# convert the standard deviation threshold from mg to g
	std_mg_threshold /= 1000
	# convert the value range threshold from mg to g
	value_range_mg_threshold /= 1000

	# new array to record non-wear time. Convention is 0 = non-wear time, and 1 = wear time. Since we create a new array filled with ones, we only have to 
	# deal with non-wear time (0), since everything else is already encoded as wear-time (1)
	non_wear_vector = np.ones((data.shape[0], 1), dtype = 'uint8')

	# loop over the data, start from the beginning with a step size of window overlap
	for i in range(0, len(data), window_overlap):

		# define the start of the sequence
		start = i
		# define the end of the sequence
		end = i + min_non_wear_time_window

		# slice the data from start to end
		subset_data = data[start:end]

		# check if the data sequence has been exhausted, meaning that there are no full windows left in the data sequence (this happens at the end of the sequence)
		# comment out if you want to use all the data
		if len(subset_data) < min_non_wear_time_window:
			break

		# calculate the standard deviation of each column (YXZ)
		std = np.std(subset_data, axis=0)

		# check if the standard deviation is below the threshold, and if the number of axes the standard deviation is below equals the std_min_num_axes threshold
		if (std < std_mg_threshold).sum() >= std_min_num_axes:

			# at least 'std_min_num_axes' are below the standard deviation threshold of 'std_min_num_axes', now set this subset of the data to 0 which will 
			# record it as non-wear time. Note that the full 'new_wear_vector' is pre-populated with all ones, so we only have to set the non-wear time to zero
			non_wear_vector[start:end] = 0

		# calculate the value range (difference between the min and max) (here the point-to-point numpy method is used) for each column
		value_range = np.ptp(subset_data, axis = 0)

		# check if the value range, for at least 'value_range_min_num_axes' (e.g. 2) out of three axes, was less than 'value_range_mg_threshold' (e.g. 50) mg
		if (value_range < value_range_mg_threshold).sum() >= value_range_min_num_axes:

			# set the non wear vector to non-wear time for the start to end slice of the data
			# Note that the full array starts with all ones, we only have to set the non-wear time to zero
			non_wear_vector[start:end] = 0

	return non_wear_vector