# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:53:57 2023

@author: Joren B., mllopis
"""
##############################################
###                                        ### 
###   ExpoApp data to Actigraph-like data  ###
###           (Python function)            ###
###                                        ###
##############################################




##########################
### Function arguments ###
##########################


# path (string): path to raw ExpoApp data, acceleration in m/s2 (unequal sampling time, more or less 30 samples/sec)
# gap_size (int): maximal duration of a gap in the data that should be interpolated, in seconds




#############################################
### Example of first rows of ExpoApp data ###
#############################################
	
# timestamp;x;y;z (it also works if this row is not omitted)
# 1638259260008;-7.3071036;-2.3175871;6.2632318
# 1638259260027;-7.335834;-2.327164;6.244078
# 1638259260048;-7.3166804;-2.27928;6.1866174
# 1638259260069;-7.374141;-2.3750482;6.2728086
# 1638259260088;-7.3645644;-2.384625;6.291962


###############
### Example ###
###############


# Create dataframe
# data_act = expoapp_to_actigraph("C:/my_path/expoapp_data.csv", 0.1)


# Save dataframe
# data_act.to_csv("C:/my_path/expoapp_data_act.csv",index=False, header=False)


def expoapp_to_actigraph(path, gap_size, time_zone):
    
    # Read raw data
    try:
        timeseries = pd.read_csv(path, sep=';', index_col='timestamp')
    except ValueError:
        timeseries = pd.read_csv(path, sep=';', names=['timestamp', 'x', 'y', 'z'], index_col='timestamp')
    
    # Transform acceleration data into g values and round to 3 digits after the comma
    timeseries[:] = timeseries[:]/9.80665  
    
    # Create datetime index in the local time zone
    timeseries.index = pd.DatetimeIndex(timeseries.index*1000000, tz = "UTC").tz_convert(time_zone)


    # Round datetime values to 1 hunderth of a second
    timeseries.index = timeseries.index.round('0.01s')
    
    # Remove duplicate datetime values and resample datetime to sampling frequency
    timeseries = timeseries[~timeseries.index.duplicated(keep='first')].asfreq('0.01s')
    
    # Start timeseries at rounded second (e.g. 2020-06-02 11:50:18.000)
    datetime_first_rounded_sec = (timeseries.index[0].strftime('%Y-%m-%d') + ' ' + 
                                  timeseries.index[0].strftime('%H:%M:') + str(timeseries.index[0].second+1))
    timeseries = timeseries[timeseries.index >= datetime_first_rounded_sec]


    # Create mask that is False if data points are more than 'gap_size' seconds apart
    mask = timeseries.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum()) 
    grp['ones'] = 1
    grp = grp.groupby('x').transform('count')
    mask = (grp['ones'] <= gap_size*100) | timeseries['x'].notnull()


    # Create dataframe with interpolated g values at 100 Hz (only interpolate if mask is True, otherwise values 0)
    timeseries = timeseries.interpolate()
    timeseries.loc[~mask, :] = 0
    
    # Downsample to 30 Hz by taking value at [3,7,10,13,17,20,23,27,30,...] hundreths of a second
    mask_30hz = np.concatenate([np.array([3,7,10])+k for k in range(0,len(timeseries.index)-10,10)])
    timeseries = timeseries.iloc[mask_30hz,:]
    timeseries = timeseries.round(3)
    
    # Make data similar to Actigraph files
    timeseries.columns = "Accelerometer X","Accelerometer Y","Accelerometer Z"
    header = [("------------ Data File Created By Joren Buekers, Sarah Koch and Maria LLopis date format "
              "dd/MM/yyyy at 30 Hz -----------"),
               "Serial Number: XXX",
               "Start Time " + timeseries.index[0].strftime('%H:%M:%S'),
               "Start Date " + timeseries.index[0].strftime('%d/%m/%Y'),
               "Epoch Period (hh:mm:ss) 00:00:00",
               "",
               "",
               "",
               "",
               "--------------------------------------------------"]
    df1 = pd.DataFrame(columns=timeseries.columns, index=range(11))
    df1.iloc[10,:] = df1.columns
    timeseries = pd.concat([df1,timeseries], ignore_index=True)
    timeseries.loc[0:9,'Accelerometer X'] = header
    
    return timeseries
