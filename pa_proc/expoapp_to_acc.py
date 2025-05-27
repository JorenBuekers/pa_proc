import pandas as pd
import numpy as np

def expoapp_to_equidistant(path, gap_size, time_zone):
    """
    ##########################
    ### Function arguments ###
    ##########################

    path: string
        path to raw ExpoApp data, acceleration in m/s2 (unequal sampling time, more or less 30 samples/sec)
    gap_size: int (optional, default = 1)
        maximal duration of a gap in the data that should be interpolated, in seconds
    time_zone: string (optional, default = "Europe/Madrid"))
        local time zone

    ##############
    ### Output ###
    ##############

    data_acceleration: pandas dataframe
        acceleration at 30 Hz in 3 axes:
            - v = vertical (x axis of of the phone)
            - ml = medio-lateral (y axis of of the phone)
            - ap = anterior-posterior (z axis of of the phone)

    #############################################
    ### Example of first rows of ExpoApp data ###
    #############################################

    timestamp;x;y;z (it also works if this row is not omitted)
    1638259260008;-7.3071036;-2.3175871;6.2632318
    1638259260027;-7.335834;-2.327164;6.244078
    1638259260048;-7.3166804;-2.27928;6.1866174
    1638259260069;-7.374141;-2.3750482;6.2728086
    1638259260088;-7.3645644;-2.384625;6.291962 
    """

    # Read raw data
    try:
        data_acceleration = pd.read_csv(path, sep=';', index_col='timestamp')
    except ValueError:
        data_acceleration = pd.read_csv(path, sep=';', names=['timestamp', 'x', 'y', 'z'], index_col='timestamp')
    
    # Transform acceleration data into g values and round to 3 digits after the comma
    data_acceleration[:] = data_acceleration[:]/9.80665  
    
    # Create datetime index in the local time zone
    data_acceleration.index = pd.DatetimeIndex(data_acceleration.index*1000000, tz = "UTC").tz_convert(time_zone)

    # Round datetime values to 1 hunderth of a second
    data_acceleration.index = data_acceleration.index.round('0.01s')
    
    # Remove duplicate datetime values and resample datetime to sampling frequency
    data_acceleration = data_acceleration[~data_acceleration.index.duplicated(keep='first')].asfreq('0.01s')
    
    # Start data_acceleration at rounded second (e.g. 2020-06-02 11:50:18.000)
    datetime_first_rounded_sec = (data_acceleration.index[0].strftime('%Y-%m-%d') + ' ' + 
                                  data_acceleration.index[0].strftime('%H:%M:') + str(data_acceleration.index[0].second+1))
    data_acceleration = data_acceleration[data_acceleration.index >= datetime_first_rounded_sec]

    # Create mask that is False if data points are more than 'gap_size' seconds apart
    mask = data_acceleration.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum()) 
    grp['ones'] = 1
    grp = grp.groupby('x').transform('count')
    mask = (grp['ones'] <= gap_size*100) | data_acceleration['x'].notnull()

    # Create dataframe with interpolated g values at 100 Hz (only interpolate if mask is True, otherwise values 0)
    data_acceleration = data_acceleration.interpolate()
    data_acceleration.loc[~mask, :] = 0
    
    # Downsample to 30 Hz by taking value at [3,7,10,13,17,20,23,27,30,...] hundreths of a second
    mask_30hz = np.concatenate([np.array([3,7,10])+k for k in range(0,len(data_acceleration.index)-10,10)])
    data_acceleration = data_acceleration.iloc[mask_30hz,:]
    data_acceleration = data_acceleration.round(3)
    
    # Update column names and add header
    data_acceleration.columns = "v","ml","ap"
    header = ["Start Time " + data_acceleration.index[0].strftime('%H:%M:%S'),
              "Start Date " + data_acceleration.index[0].strftime('%d/%m/%Y')]
    df1 = pd.DataFrame(columns=data_acceleration.columns, index=range(3))
    df1.iloc[2,:] = df1.columns
    data_acceleration = pd.concat([df1,data_acceleration], ignore_index=True)
    data_acceleration.loc[0:1,'v'] = header
    
    return data_acceleration

def expoapp_to_actigraph(path, gap_size, time_zone):
    """
    ##########################
    ### Function arguments ###
    ##########################

    path: string
        path to raw ExpoApp data, acceleration in m/s2 (unequal sampling time, more or less 30 samples/sec)
    gap_size: int (optional, default = 1)
        maximal duration of a gap in the data that should be interpolated, in seconds
    time_zone: string (optional, default = "Europe/Madrid"))
        local time zone

    ##############
    ### Output ###
    ##############

    data_actigraph: pandas dataframe
        acceleration at 30 Hz in 3 axes:
            - v = vertical (x axis of of the phone)
            - ml = medio-lateral (y axis of of the phone)
            - ap = anterior-posterior (z axis of of the phone)
    
    #############################################
    ### Example of first rows of ExpoApp data ###
    #############################################

    timestamp;x;y;z (it also works if this row is not omitted)
    1638259260008;-7.3071036;-2.3175871;6.2632318
    1638259260027;-7.335834;-2.327164;6.244078
    1638259260048;-7.3166804;-2.27928;6.1866174
    1638259260069;-7.374141;-2.3750482;6.2728086
    1638259260088;-7.3645644;-2.384625;6.291962
    """

    # Read raw data
    try:
        data_actigraph = pd.read_csv(path, sep=';', index_col='timestamp')
    except ValueError:
        data_actigraph = pd.read_csv(path, sep=';', names=['timestamp', 'x', 'y', 'z'], index_col='timestamp')
    
    # Transform acceleration data into g values and round to 3 digits after the comma
    data_actigraph[:] = data_actigraph[:]/9.80665  
    
    # Create datetime index in the local time zone
    data_actigraph.index = pd.DatetimeIndex(data_actigraph.index*1000000, tz = "UTC").tz_convert(time_zone)

    # Round datetime values to 1 hunderth of a second
    data_actigraph.index = data_actigraph.index.round('0.01s')
    
    # Remove duplicate datetime values and resample datetime to sampling frequency
    data_actigraph = data_actigraph[~data_actigraph.index.duplicated(keep='first')].asfreq('0.01s')
    
    # Start data_actigraph at rounded second (e.g. 2020-06-02 11:50:18.000)
    datetime_first_rounded_sec = (data_actigraph.index[0].strftime('%Y-%m-%d') + ' ' + 
                                  data_actigraph.index[0].strftime('%H:%M:') + str(data_actigraph.index[0].second+1))
    data_actigraph = data_actigraph[data_actigraph.index >= datetime_first_rounded_sec]

    # Create mask that is False if data points are more than 'gap_size' seconds apart
    mask = data_actigraph.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum()) 
    grp['ones'] = 1
    grp = grp.groupby('x').transform('count')
    mask = (grp['ones'] <= gap_size*100) | data_actigraph['x'].notnull()

    # Create dataframe with interpolated g values at 100 Hz (only interpolate if mask is True, otherwise values 0)
    data_actigraph = data_actigraph.interpolate()
    data_actigraph.loc[~mask, :] = 0
    
    # Downsample to 30 Hz by taking value at [3,7,10,13,17,20,23,27,30,...] hundreths of a second
    mask_30hz = np.concatenate([np.array([3,7,10])+k for k in range(0,len(data_actigraph.index)-10,10)])
    data_actigraph = data_actigraph.iloc[mask_30hz,:]
    data_actigraph = data_actigraph.round(3)
    
    # Make data similar to Actigraph files
    data_actigraph.columns = "Accelerometer X","Accelerometer Y","Accelerometer Z"
    header = [("------------ Data File Created By Joren Buekers, Sarah Koch and Maria LLopis date format "
              "dd/MM/yyyy at 30 Hz -----------"),
               "Serial Number: XXX",
               "Start Time " + data_actigraph.index[0].strftime('%H:%M:%S'),
               "Start Date " + data_actigraph.index[0].strftime('%d/%m/%Y'),
               "Epoch Period (hh:mm:ss) 00:00:00",
               "",
               "",
               "",
               "",
               "--------------------------------------------------"]
    df1 = pd.DataFrame(columns=data_actigraph.columns, index=range(11))
    df1.iloc[10,:] = df1.columns
    data_actigraph = pd.concat([df1,data_actigraph], ignore_index=True)
    data_actigraph.loc[0:9,'Accelerometer X'] = header
    
    return data_actigraph
