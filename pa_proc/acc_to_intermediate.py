import pandas as pd
import numpy as np
import datetime
from agcounts.extract import get_counts
import skdh

from pa_proc.nonwear import weartime_choi2011
from pa_proc.nonwear import weartime_vanhees

def acc_to_intermediate(path, device, sampling_freq, autocalibration):

    ##########################
    ### Function arguments ###
    ##########################
    
    # path (string): path to acceleration data in Actigraph-like style, in g
    # device (string): indicate which device was used to collect the data: 'actigraph' or 'expoapp'
    # sampling_freq (int): sampling frequency of input dataframe, in hz
    # autocalibration (0 or 1): if 1, autocalibration will be performed based on Van Hees et al., 2013 for the ENMO pathway
    
    ################
    ### Function ###
    ################

    if device == 'expoapp-csv':
        header = pd.read_csv(path, nrows=2, header=None)
        start_date = header.iloc[1,0][-10:].strip()
        start_second = np.int32(header.iloc[0,0][-2:].strip())
        start_minute = np.int32(header.iloc[0,0][-5:-3].strip())
        start_time = header.iloc[0,0][-8:-5].strip() + "{:02d}".format(start_minute) + ':00' 
        start_datetime = (datetime.datetime.strptime(start_date + ' ' + start_time, '%d/%m/%Y %H:%M:%S').
                        strftime('%Y-%m-%d %H:%M:%S'))
        samples_to_skip_until_first_rounded_minute = 0 if (60-start_second)*30==1800 else (60-start_second)*30 # start at first rounded minute

        timeseries = pd.read_csv(path, 
                                 skiprows = 3 + samples_to_skip_until_first_rounded_minute,
                                 names=['v', 'ml', 'ap']) #The axis order is vertical, horizontal and perpendicular
    
    elif device == 'actigraph-csv':

        # Extract datetime information from the Actigraph-like file
        header = pd.read_csv(path, nrows=10, header=None)
        start_date = header.iloc[3,0][-10:].strip()
        start_second = np.int32(header.iloc[2,0][-2:].strip())
        start_minute = np.int32(header.iloc[2,0][-5:-3].strip())
        start_time = header.iloc[2,0][-8:-5].strip() + "{:02d}".format(start_minute) + ':00' 
        start_datetime = (datetime.datetime.strptime(start_date + ' ' + start_time, '%d/%m/%Y %H:%M:%S').
                        strftime('%Y-%m-%d %H:%M:%S'))
        samples_to_skip_until_first_rounded_minute = 0 if (60-start_second)*30==1800 else (60-start_second)*30 # start at first rounded minute

        timeseries = pd.read_csv(path,  sep = ' ',
                             skiprows= 11 + samples_to_skip_until_first_rounded_minute, 
                             names=['xyz']) # Specify the separator so everything will be loaded into 1 column. ADAPT TO STUDY (e.g. actilife uses dots as separators).
        timeseries = timeseries['xyz'].str.split(',', expand=True)
        timeseries = timeseries.rename(columns={0: "ml", 1: "v", 2: "ap"}) #The axis order is horizontal, vertical and perpendicular
        timeseries = timeseries.astype(float)
        timeseries = timeseries[['v', 'ml', 'ap']] # Change order of axes
    
    elif device == 'actigraph-gt3x': 
        from pygt3x.reader import FileReader
        with FileReader(path) as reader:
            timeseries = reader.to_pandas()
        timeseries.index = pd.to_datetime(timeseries.index, unit='s')
        start_datetime = str(timeseries.index[0])
        start_second = np.int32(str(timeseries.index[0])[-2:])
        samples_to_skip_until_first_rounded_minute = 0 if (60-start_second)*30==1800 else (60-start_second)*30 # start at first rounded minute
        timeseries.reset_index(inplace=True)
        timeseries = timeseries.rename(columns={"X": "ml", "Y": "v", "Z": "ap"})
        timeseries = timeseries[['v', 'ml', 'ap']] # Change order of axes and drop unnecesarry columns
    
    # Calculate counts for the different axes and combine
    counts = get_counts(np.array(timeseries), freq=sampling_freq, epoch=60)
    counts = counts = pd.DataFrame(counts, columns=['counts_v', 'counts_ml', 'counts_ap'])  #The axis correspond to 1=vertical, 2=horizontal and 3=perpendicular
    counts['counts_vm'] = np.sqrt(np.sum(np.square(np.array(counts)), axis=1))
    counts.index = np.datetime64(start_datetime) + np.arange(0,len(counts.index)*60,60)

    # Apply choi algorithm to vector magnitude of Actigraph counts
    counts['wear_choi'] = weartime_choi2011(np.array(counts[["counts_v","counts_ml","counts_ap"]]), 
                                            np.array(counts.index),
                                            use_vector_magnitude = True)
    
    # Auto-calibration
    if autocalibration == 1: 
        skdh_pipeline = skdh.Pipeline()
        skdh_pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
        calibrated_dict = skdh_pipeline.run(time=timeseries.index, accel= timeseries.values)
        timeseries = pd.DataFrame(calibrated_dict.get('CalibrateAccelerometer').get('accel'), 
                                  columns=['v','ml','ap'])
         
	# Sensor wear based on van Hees 2013
    timeseries['wear_vanhees'] = weartime_vanhees(np.array(timeseries[['v', 'ml', 'ap']]),
                                                  hz = sampling_freq).astype(int)

    # Calculate ENMO
    enmo = np.sqrt(np.sum(np.square(timeseries[['v', 'ml', 'ap']]), axis=1)) - 1
    enmo = enmo.clip(lower=0) # Set negative values to 0
    enmo *= 1000 # Multiply by 1000 to get mg

    # Downsample to 1 Hz and add info on wear based on van Hees 2011
    enmo_1s = pd.DataFrame({'enmo': enmo.groupby(enmo.index // sampling_freq).mean(), 
                           'wear_vanhees': timeseries['wear_vanhees'].groupby(timeseries.index // sampling_freq).min()}) 
    enmo_1s.index = np.datetime64(start_datetime) + np.arange(len(enmo_1s.index))

    # Final dataframes at 10s and 60s epochs (by averaging ENMO)
    enmo_10s = enmo_1s.resample('10s').mean()
    enmo_10s.wear_vanhees = enmo_10s.wear_vanhees.apply(np.floor)

    enmo_60s = enmo_1s.resample('60s').mean()
    enmo_60s.wear_vanhees = enmo_60s.wear_vanhees.apply(np.floor)

    intermediate_60s = pd.concat([counts,enmo_60s], axis=1)
    
    return intermediate_60s, enmo_10s
