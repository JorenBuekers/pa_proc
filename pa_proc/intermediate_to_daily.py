
import pandas as pd
import numpy as np


def intermediate_to_daily(intermediate, cutpoints_counts = [], cutpoints_enmo = [], waking_hours = [7, 22]):
    """
    ##########################
    ### Function arguments ###
    ##########################
    
    intermediate: Pandas dataframe (1 value per minute)
        intermediate variables obtained from acc_to_intermediate
    cutpoints_counts: list of 3 integers
        list containing the Actigraph count cutpoints for:
            - low intensity physical activity
            - moderate intensity physical activity
            - vigorous intensity physical activity
    cutpoints_enmo: list of 3 integers
       list containing the ENMO cutpoints for:
            - low intensity physical activity
            - moderate intensity physical activity
            - vigorous intensity physical activity
    waking_hours: list of 2 integers (optional)
        list containing borders of waking hours in the local timezone (default = 07h00-22h00)

    ##############
    ### Output ###
    ##############

    pa_daily: pandas dataframe (1 value per day)
        - weartime_choi: wear time from midnight to midnight (in minutes), as estimated by the algorithm of Choi et al., 2011
        - weartime_vanhees: wear time from midnight to midnight (in minutes), as estimated by the algorithm of Choi et al., 2011
        - weartime_choi_waking: wear time during waking hours (in minutes), as estimated by the algorithm of Choi et al., 2011
        - weartime_vanhees_waking:  wear time during waking hours (in minutes), as estimated by the algorithm of Choi et al., 2011
        - lpa_counts: time spent in low intensity physical activity (in minutes), based on Actigraph counts
        - mpa_counts: time spent in moderate intensity physical activity (in minutes), based on Actigraph counts
        - mvpa_counts: time spent in moderate-to-vigorous intensity physical activity (in minutes), based on Actigraph counts
        - vpa_counts: time spent in vigorous intensity physical activity (in minutes), based on Actigraph counts
        - lpa_ENMO: time spent in low intensity physical activity (in minutes), based on ENMO
        - mpa_ENMO: time spent in moderate intensity physical activity (in minutes), based onENMO
        - mvpa_ENMO: time spent in moderate-to-vigorous intensity physical activity (in minutes), based on ENMO
        - vpa_ENMO: time spent in vigorous intensity physical activity (in minutes), based on ENMO
       """
    # Identify different days in data
    days_measured = intermediate.index.map(pd.Timestamp.date).unique()
    
    # Create pa_daily dataframe
    pa_daily = pd.DataFrame(index=days_measured)
    
    # Calculate time spent in different physical activity intensities, based on Actigraph counts and/or ENMO
    for day_loop in days_measured:
        
        # Extract data of examined day
        day_data = intermediate[(intermediate.index.date == day_loop)]

        # Extract data of examined day during waking hours
        day_data_waking = day_data[(day_data.index.hour >= waking_hours[0]) & 
                                   (day_data.index.hour < waking_hours[1])]
        
        # Determine daily wear time (all data and during waking hours only)
        pa_daily.loc[day_loop, 'weartime_choi'] = np.sum(day_data.wear_choi == 1)
        pa_daily.loc[day_loop, 'weartime_vanhees'] = np.sum(day_data.wear_vanhees == 1)

        pa_daily.loc[day_loop, 'weartime_choi_waking'] = np.sum(day_data_waking.wear_choi == 1)
        pa_daily.loc[day_loop, 'weartime_vanhees_waking'] = np.sum(day_data_waking.wear_vanhees == 1)
        
        # Select data when device is worn
        day_data_wear_choi = day_data[day_data.wear_choi == 1]
        day_data_wear_vanhees = day_data[day_data.wear_vanhees == 1]
        
        # Calculate time spent in different physical activity intensities
        if cutpoints_counts: # if list is not empty
            pa_daily.loc[day_loop, 'lpa_counts'] = np.sum((day_data_wear_choi.counts_vm > cutpoints_counts[0]) & 
                                                    (day_data_wear_choi.counts_vm <= cutpoints_counts[1]))
            pa_daily.loc[day_loop, 'mpa_counts'] = np.sum((day_data_wear_choi.counts_vm > cutpoints_counts[1]) & 
                                                    (day_data_wear_choi.counts_vm <= cutpoints_counts[2]))
            pa_daily.loc[day_loop, 'mvpa_counts'] = np.sum(day_data_wear_choi.counts_vm > cutpoints_counts[1])
            pa_daily.loc[day_loop, 'vpa_counts'] = np.sum(day_data_wear_choi.counts_vm > cutpoints_counts[2])

        if cutpoints_enmo:# if list is not empty
            pa_daily.loc[day_loop, 'lpa_enmo'] = np.sum((day_data_wear_vanhees.enmo > cutpoints_enmo[0])
                                                & (day_data_wear_vanhees.enmo < cutpoints_enmo[1]))
            pa_daily.loc[day_loop, 'mpa_enmo'] = np.sum((day_data_wear_vanhees.enmo > cutpoints_enmo[1])
                                                & (day_data_wear_vanhees.enmo < cutpoints_enmo[2]))
            pa_daily.loc[day_loop, 'mvpa_enmo'] = np.sum(day_data_wear_vanhees.enmo >= cutpoints_enmo[1])
            pa_daily.loc[day_loop, 'vpa_enmo'] = np.sum(day_data_wear_vanhees.enmo >= cutpoints_enmo[2])

    return pa_daily