###############################################
###                                         ###
###   Acceleration data to counts and ENMO  ###
###       	(Python function)         	    ###
###                                     	###
############################################### 
 
##########################
### Function arguments ###
##########################
 
# path (string): path to acceleration data in Actigraph-like style, in g
# device (string): indicate which device was used to collect the data: 'actigraph' or 'expoapp'
# sampling_freq (int): sampling frequency of input dataframe, in hz
# autocalibration (0 or 1): if 1, autocalibration will be performed based on Van Hees et al., 2013 for the ENMO pathway
 
###############
### Example ###
###############
 
# Create dataframe
# [counts_enmo_10s, counts_enmo_60s] = acc_to_counts_enmo("C:/my_path/actigraph_like_data.csv", 30)
 
#########################
### Required imports  ###
#########################
 
import pandas as pd
import numpy as np
import datetime
from agcounts.extract import get_counts
import skdh


def acc_to_intermediate(path, device, sampling_freq, autocalibration):
    from pa_proc.nonwear import weartime_choi2011
    from pa_proc.nonwear import weartime_vanhees
	

    # Read raw data (slightly different for Actigraph and ExpoApp)
    if device == 'actigraph':

        # Extract datetime information from the Actigraph-like file
        header = pd.read_csv(path, nrows=10, header=None)
        start_date = header.iloc[3,0][-10:].strip()
        start_second = np.int32(header.iloc[2,0][-2:].strip())
        start_minute = np.int32(header.iloc[2,0][-5:-3].strip())
        start_time = header.iloc[2,0][-8:-5].strip() + "{:02d}".format(start_minute) + ':00' # start at first rounded minute
        start_datetime = (datetime.datetime.strptime(start_date + ' ' + start_time, '%d/%m/%Y %H:%M:%S').
                        strftime('%Y-%m-%d %H:%M:%S'))
        samples_to_skip_until_first_rounded_minute = (60-start_second)*30

        timeseries = pd.read_csv(path,  sep = ' ',
                             skiprows= 11 + samples_to_skip_until_first_rounded_minute, 
                             names=['xyz']) # Specify the separator so everything will be loaded into 1 column. ADAPT TO STUDY (e.g. actilife uses dots as separators).
        timeseries = timeseries['xyz'].str.split(',', expand=True)
        timeseries = timeseries.rename(columns={0: "ml", 1: "v", 2: "ap"}) #The axis order is horizontal, vertical and perpendicular
        timeseries = timeseries.astype(float)
    elif device == 'expoapp':
        timeseries = pd.read_csv(path, 
                             skiprows= 11 + samples_to_skip_until_first_rounded_minute, 
                             names=['v', 'ml', 'ap']) #The axis order is vertical, horizontal and perpendicular  
    
    
    # Calculate counts for the different axes and combine
    counts = get_counts(np.array(timeseries[['v', 'ml', 'ap']]), freq=sampling_freq, epoch=60)
    counts = pd.DataFrame(counts, columns=['v', 'ml', 'ap'])  #The axis correspond to 1=vertical, 2=horizontal and 3=perpendicular
    counts['counts_vm'] = np.sqrt(counts.Axis1 **2 + counts.Axis2 **2 + counts.Axis3 **2)
    counts.index = np.datetime64(start_datetime) + np.arange(0,len(counts.index)*60,60)
    
    # Auto-calibration
    if autocalibration == 1: 
        skdh_pipeline = skdh.Pipeline()
        skdh_pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
        calibrated_dict = skdh_pipeline.run(time=timeseries.index, accel= timeseries.values)
        timeseries = pd.DataFrame(calibrated_dict.get('CalibrateAccelerometer').get('accel'), 
                                  columns=['ml','v','ap'])
 
        
	# Sensor wear based on van Hees 2013
    timeseries['wear_vanhees'] = weartime_vanhees(timeseries[['v', 'ml', 'ap']].to_numpy(),
                                                  hz = sampling_freq)


    # Calculate ENMO
    enmo =  (np.sqrt(timeseries.v **2 + timeseries.h **2 + timeseries.p **2) - 1)*1000 # Multiply by 1000 to get mg
    enmo = enmo.clip(lower=0) # Correct for when there is no data (value of -1000) or negative values (as done by van Hees)


    # Downsample to 1 Hz and add info on wear based on van Hees 2011
    enmo_1s = pd.DataFrame({'enmo': enmo.groupby(enmo.index // sampling_freq).mean(), 
                           'wear_vanhees': timeseries['wear_vanhees'].groupby(timeseries.index // sampling_freq).min()}) 
    enmo_1s.index = np.datetime64(start_datetime) + np.arange(len(enmo_1s.index))

    # Final dataframes at 10s and 60s epochs (by summing counts and averaging ENMO)
    enmo_10s = enmo_1s.resample('10s').mean()
    enmo_10s.wear_vanhees = enmo_10s.wear_vanhees.apply(np.floor)

    enmo_60s = enmo_1s.resample('60s').mean()
    enmo_60s.wear_vanhees = enmo_60s.wear_vanhees.apply(np.floor)

    counts_enmo = pd.concat([counts,enmo_60s], axis=1)

    # Non-wear based on Choi 2011
    wear_data_choi = counts_enmo[['Axis1','Axis2','Axis3']].to_numpy()
    
    # Apply choi algorithm to all 3 axes of count data, to deal with weird performance of agcounts at some points
    choi_x = weartime_choi2011(wear_data_choi, counts_enmo.index, 0)
    choi_y = weartime_choi2011(wear_data_choi, counts_enmo.index, 1)
    choi_z = weartime_choi2011(wear_data_choi, counts_enmo.index, 2)
    counts_enmo['wear_choi'] = np.floor((choi_x + choi_y + choi_z)/3)
    
    
    return enmo_10s, counts_enmo