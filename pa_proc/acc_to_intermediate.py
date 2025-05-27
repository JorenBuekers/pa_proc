import pandas as pd
import numpy as np
import datetime
from agcounts.extract import get_counts
import skdh

from pa_proc.nonwear import weartime_choi2011
from pa_proc.nonwear import weartime_vanhees2013

def acc_to_intermediate(path, device, sampling_freq, autocalibration, location):
    """
    ##########################
    ### Function arguments ###
    ##########################
    
    path: string
        path to acceleration data in 3 axes, in g
    device: string (expoapp-csv, actigraph-csv, actigraph-gt3x, axivity-cwa or matrix-csv)
        indicate which device was used to collect the data, and in which format the file was stored
    sampling_freq: int
        sampling frequency of input dataframe, in Hz
    autocalibration: 0 or 1
        if 1, autocalibration will be performed for the ENMO pathway, based on Van Hees et al., 2013 
    location: string (lumbar, hip or wrist)
        location where the device was worn

    ##############
    ### Output ###
    ##############

    intermediate_60s: pandas dataframe (1 value per minute)
        - counts_v: Actigraph counts for the vertical axis
        - counts_ml: Actigraph counts for the mediolateral axis
        - counts_ap: Actigraph counts for the anterioposterior axis
        - counts_vm: Actigraph counts for the vector magnitude of the three axis
        - wear_choi: The device was worn (1) or not worn (0), based on the algorithm of Choi et al., 2011 (implementation from Syed et al., 2020)
        - enmo: Eucledian Norm Minues One values
        - wear_vanhees: The device was worn (1) or not worn (0), based on the algorithm of Van Hees et al., 2013 (implementation from Syed et al., 2020)

    enmo_10s: pandas dataframe (1 value per 10 seconds)
        - enmo: Eucledian Norm Minues One values
        - wear_vanhees: The device was worn (1) or not worn (0), based on the algorithm of Van Hees et al., 2013 (implementation from Syed et al., 2020)
    """

    if device == 'expoapp-csv':

        # Extract datetime information
        header = pd.read_csv(path, nrows=2, header=None)
        start_date = header.iloc[1,0][-10:].strip()
        start_second = np.int32(header.iloc[0,0][-2:].strip()) # (data before the first rounded second was already removed in expoapp_to_acc.py)
        start_minute = np.int32(header.iloc[0,0][-5:-3].strip())
        start_time = header.iloc[0,0][-8:-5].strip() + "{:02d}".format(start_minute) + ':00' 
        start_datetime = (datetime.datetime.strptime(start_date + ' ' + start_time, '%d/%m/%Y %H:%M:%S').
                        strftime('%Y-%m-%d %H:%M:%S'))
        
        # Identify location of first rounded minute (data of the incomplete minute before will be deleted)
        loc_first_round_minute = 0 if (60-start_second)*sampling_freq==1800 else (60-start_second)*sampling_freq # start at first rounded minute

        # Load acceleration time series 
        timeseries = pd.read_csv(path, 
                                 skiprows = 3 + loc_first_round_minute, # start at first rounded minute
                                 names=['v', 'ml', 'ap']) # The axis order is vertical, mediolateral and anterioposterior
    

    elif device == 'actigraph-csv':

        # Extract datetime information
        header = pd.read_csv(path, nrows=10, header=None)
        start_date = header.iloc[3,0][-10:].strip()
        start_second = np.int32(header.iloc[2,0][-2:].strip()) # (actigraph csv files always start at first rounded second)
        start_minute = np.int32(header.iloc[2,0][-5:-3].strip())
        start_time = header.iloc[2,0][-8:-5].strip() + "{:02d}".format(start_minute) + ':00' 
        start_datetime = (datetime.datetime.strptime(start_date + ' ' + start_time, '%d/%m/%Y %H:%M:%S').
                        strftime('%Y-%m-%d %H:%M:%S'))
        
        # Identify location of first rounded minute (data of the incomplete minute before will be deleted)
        loc_first_round_minute = 0 if (60-start_second)*sampling_freq==1800 else (60-start_second)*sampling_freq 

        # Load acceleration time series 
        timeseries = pd.read_csv(path,  sep = ' ',
                             skiprows= 11 + loc_first_round_minute, # start at first rounded minute
                             names=['xyz']) # Specify the separator so everything will be loaded into 1 column. 
        timeseries = timeseries['xyz'].str.split(',', expand=True)
        timeseries = timeseries.rename(columns={0: "ml", 1: "v", 2: "ap"}) # The axis order is mediolateral, vertical and anterioposterior
        timeseries = timeseries.astype(float)
        timeseries = timeseries[['v', 'ml', 'ap']] # Change order of axes
    
    elif device == 'actigraph-gt3x': 
        
        # Load acceleration time series 
        from pygt3x.reader import FileReader
        with FileReader(path) as reader:
            timeseries = reader.to_pandas()
        timeseries.index = pd.to_datetime(timeseries.index, unit='s')

        # Identify location of first rounded minute (data of the incomplete minute before will be deleted)
        first_round_minute = timeseries.loc[timeseries.index.minute==timeseries.index[0].minute+1].index[0]
        loc_first_round_minute = timeseries.index.get_loc(first_round_minute)
        loc_first_round_minute = 0 if loc_first_round_minute==sampling_freq*60 else loc_first_round_minute

        # Remove data before first rounded minute
        timeseries = timeseries[loc_first_round_minute:]

        # Extract datetime information
        start_datetime = timeseries.index[0]
        start_second = start_datetime.second

        # Adapt index and column names
        timeseries.reset_index(inplace=True)
        timeseries = timeseries.rename(columns={"X": "ml", "Y": "v", "Z": "ap"}) # The axis order is mediolateral, vertical and anterioposterior
        timeseries = timeseries[['v', 'ml', 'ap']] # Change order of axes and drop unnecesarry columns
    
    elif device == 'axivity-cwa': 

        # Load acceleration time series 
        from openmovement.load import CwaData
        with CwaData(path, include_gyro=False, include_temperature=False) as cwa_data:
            timeseries = cwa_data.get_samples()
        timeseries.set_index("time", inplace=True)
        
        # Identify location of first rounded minute (data of the incomplete minute before will be deleted)
        first_round_minute = timeseries.loc[timeseries.index.minute==timeseries.index[0].minute+1].index[0]
        loc_first_round_minute = timeseries.index.get_loc(first_round_minute)
        loc_first_round_minute = 0 if loc_first_round_minute==sampling_freq*60 else loc_first_round_minute

        # Remove data before first rounded minute
        timeseries = timeseries[loc_first_round_minute:]

        # Extract datetime information
        start_datetime = timeseries.index[0]
        start_second = start_datetime.second
        
        # Adapt index and column names
        timeseries.reset_index(inplace=True)
        timeseries = timeseries.rename(columns={"accel_x": "v", "accel_y": "ml", "accel_z": "ap"}) # The axis order is vertical, mediolateral and anterioposterior
        timeseries = timeseries[['v', 'ml', 'ap']] # Change order of axes and drop unnecesarry columns

    elif device == 'matrix-csv': 
        
        # Load acceleration time series
        data_matrix = pd.read_csv(path, index_col='dateTime', usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
        
        # Make into an equidistant time series
        data_matrix.index = pd.DatetimeIndex(data_matrix.index, tz = "Europe/Madrid")
        if sampling_freq==100:
            timeseries = data_matrix.resample("0.01s").ffill() 
        else:
            print("Code still needs to be added for when sampling frequency of the Matrix device is not 100 Hz")

        # Identify location of first rounded minute (data of the incomplete minute before will be deleted)
        first_round_minute = timeseries.loc[timeseries.index.minute==timeseries.index[0].minute+1].index[0]
        loc_first_round_minute = timeseries.index.get_loc(first_round_minute)
        loc_first_round_minute = 0 if loc_first_round_minute==sampling_freq*60 else loc_first_round_minute

        # Remove data before first rounded minute
        timeseries = timeseries[loc_first_round_minute:]

        # Extract datetime information
        start_datetime = timeseries.index[0]
        start_second = start_datetime.second

        # Adapt index and column names
        timeseries.reset_index(inplace=True)
        timeseries = timeseries.rename(columns={"acc_x": "v", "acc_y": "ml", "acc_z": "ap"}) # Rename for running the code, but reverse naming at the end of script
        timeseries = timeseries[['v', 'ml', 'ap']] # Drop unnecesarry columns

    # Calculate counts for the different axes and combine
    counts = get_counts(np.array(timeseries), freq=sampling_freq, epoch=60)
    counts = pd.DataFrame(counts, columns=['counts_v', 'counts_ml', 'counts_ap'])  #The axis correspond to 1=vertical, 2=horizontal and 3=perpendicular
    counts['counts_vm'] = np.sqrt(np.sum(np.square(np.array(counts)), axis=1))
    counts.index = np.datetime64(start_datetime, "s") + np.arange(0,len(counts.index)*60,60)

    # Apply choi algorithm to vector magnitude of Actigraph counts
    counts['wear_choi'] = weartime_choi2011(np.array(counts[["counts_v","counts_ml","counts_ap"]]), 
                                            np.array(counts.index),
                                            use_vector_magnitude = True)
    
    # Auto-calibration
    if autocalibration == 1: 
        skdh_pipeline = skdh.Pipeline()
        skdh_pipeline.add(skdh.preprocessing.CalibrateAccelerometer())
        calibrated_dict = skdh_pipeline.run(time=timeseries.index, accel= timeseries.values)
        
        # Only if auto-calibration was performed, obtained calibrated values (otherwise calibrated_dict is empty)
        if len(calibrated_dict.get('CalibrateAccelerometer')) != 0: 
            timeseries = pd.DataFrame(calibrated_dict.get('CalibrateAccelerometer').get('accel'), 
                                        columns=['v','ml','ap'])
         
	# Sensor wear based on van Hees 2013
    timeseries['wear_vanhees'] = weartime_vanhees2013(np.array(timeseries[['v', 'ml', 'ap']]),
                                                  hz = sampling_freq).astype(int)

    # Calculate ENMO
    enmo = np.sqrt(np.sum(np.square(timeseries[['v', 'ml', 'ap']]), axis=1)) - 1
    enmo = enmo.clip(lower=0) # Set negative values to 0
    enmo *= 1000 # Multiply by 1000 to get mg

    # Downsample to 1 Hz and add info on wear based on van Hees 2011
    enmo_1s = pd.DataFrame({'enmo': enmo.groupby(enmo.index // sampling_freq).mean(), 
                           'wear_vanhees': timeseries['wear_vanhees'].groupby(timeseries.index // sampling_freq).min()}) 
    enmo_1s.index = np.datetime64(start_datetime, "s") + np.arange(len(enmo_1s.index))

    # Final dataframes at 10s and 60s epochs (by averaging ENMO)
    enmo_10s = enmo_1s.resample('10s').mean()
    enmo_10s.wear_vanhees = enmo_10s.wear_vanhees.apply(np.floor)

    enmo_60s = enmo_1s.resample('60s').mean()
    enmo_60s.wear_vanhees = enmo_60s.wear_vanhees.apply(np.floor)
    intermediate_60s = pd.concat([counts,enmo_60s], axis=1)

    # If the device is worn on the wrist, change back naming of axes
    if location == "wrist":
        intermediate_60s = intermediate_60s.rename(columns={"counts_v": "counts_x", 
                                                            "counts_ml": "counts_y", 
                                                            "counts_ap": "counts_z"})
   
    return intermediate_60s, enmo_10s
