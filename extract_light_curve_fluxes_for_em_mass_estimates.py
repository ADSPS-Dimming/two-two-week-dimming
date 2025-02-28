import os
from scipy.io import readsav
import datetime
import pandas as pd  # Import pandas
import numpy as np  # Import numpy for handling arrays
from astropy.time import Time

def load_eve_lines(event_num):
    base_path = os.path.expanduser('~/Dropbox/Research/Woods_LASP/Analysis/Coronal Dimming Analysis/Two Two Week Period/EVEPlots/Corrected/')
    event_path = os.path.join(base_path, f'Event{event_num}', 'Warm correction', 'EVELines.sav')
    
    if os.path.exists(event_path):
        data = readsav(event_path)
        
        irradiance = data['evelines']['line_irradiance']
        irradiance = convert_nested_to_2d_array(irradiance)
        
        yyyydoy = data['evelines']['yyyydoy']
        sod = data['evelines']['sod']
        time_iso = convert_to_iso8601(yyyydoy, sod)

        wavelength = data['evemeta']['linesmeta'][0]['wave_center']

        light_curve_data = {
            'time_iso': time_iso,
            'wavelength': wavelength,
            'irradiance': irradiance
        }
        return light_curve_data
    else:
        print(f"File not found: {event_path}")
        return None

def load_preflare_times(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None, names=['date_str', 'seconds_of_day'], dtype={'date_str': str})
    
    # Convert the date and time information into ISO8601 format
    preflare_times = []
    for _, row in df.iterrows():
        year = int(row['date_str'][:4])
        doy = int(row['date_str'][4:])
        seconds_of_day = int(row['seconds_of_day'])
        
        # Convert to ISO8601 format
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1, seconds=seconds_of_day)
        iso8601_timestamp = date.isoformat()
        preflare_times.append(iso8601_timestamp)
    
    return preflare_times

def convert_to_iso8601(yyyydoy_array, sod_array):
    iso8601_timestamps = []
    for yyyydoy, sod in zip(yyyydoy_array, sod_array):
        year = int(str(yyyydoy)[:4])
        doy = int(str(yyyydoy)[4:])
        seconds_of_day = int(sod)
        
        # Convert to ISO8601 format
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1, seconds=seconds_of_day)
        iso8601_timestamp = date.isoformat()
        iso8601_timestamps.append(iso8601_timestamp)
    
    return iso8601_timestamps

def convert_nested_to_2d_array(nested_array):
    """
    Convert a nested array to a 2D numpy array.
    
    Parameters:
    nested_array (list or np.ndarray): The nested array to convert.
    
    Returns:
    np.ndarray: A 2D numpy array.
    """
    # Initialize an empty list to hold the 2D array
    two_d_array = []
    
    # Loop through the nested array to construct the 2D array
    for sub_array in nested_array:
        two_d_array.append(sub_array)
    
    # Convert the list to a 2D numpy array
    return np.array(two_d_array)

def find_min_irradiance_within_window(eve_lines, preflare_time):
    # Convert preflare_time to a datetime object
    preflare_datetime = datetime.datetime.fromisoformat(preflare_time)
    end_time = preflare_datetime + datetime.timedelta(hours=4)
    
    min_irradiances = {}
    
    # Assuming eve_lines contains a dictionary with wavelengths as keys
    # and their corresponding time series data as values
    # Extract time and irradiance arrays
    times = eve_lines['time_iso']  # Assuming 'time_iso' is a key in eve_lines
    irradiances = eve_lines['irradiance']  # Assuming 'irradiance' is a key in eve_lines
    
    # Convert times to datetime objects
    times_datetime = [datetime.datetime.fromisoformat(t) for t in times]
    
    # Find indices within the specified time window
    indices_within_window = [i for i, t in enumerate(times_datetime) if preflare_datetime <= t <= end_time]
    
    if indices_within_window:
        # Find the minimum irradiance within the window for each wavelength
        for wavelength_index, wavelength in enumerate(eve_lines['wavelength']):
            min_irradiance = np.nanmin([irradiances[i, wavelength_index] for i in indices_within_window])
            min_irradiances[wavelength] = min_irradiance
    else:
        for wavelength in eve_lines['wavelength']:
            min_irradiances[wavelength] = None  # No data in the window
    return min_irradiances

def find_irradiance_at_nearest_time(eve_lines, preflare_time):
    # Convert preflare_time to a datetime object
    preflare_datetime = datetime.datetime.fromisoformat(preflare_time)
    
    # Convert times in eve_lines to datetime objects
    times_datetime = [datetime.datetime.fromisoformat(t) for t in eve_lines['time_iso']]
    
    # Sort indices by their time distance from preflare_time
    sorted_indices = sorted(range(len(times_datetime)), key=lambda i: abs(times_datetime[i] - preflare_datetime))
    
    # Create a dictionary to map wavelengths to their corresponding irradiance values
    irradiance_at_nearest_time = {}
    
    # For each wavelength, find the nearest non-NaN value
    for wavelength_index, wavelength in enumerate(eve_lines['wavelength']):
        # Try indices in order of increasing distance until a non-NaN value is found
        for idx in sorted_indices:
            irradiance_value = eve_lines['irradiance'][idx, wavelength_index]
            if not np.isnan(irradiance_value):
                irradiance_at_nearest_time[wavelength] = irradiance_value
                break
        else:
            # If all values are NaN, set to None
            irradiance_at_nearest_time[wavelength] = None
    
    return irradiance_at_nearest_time

def save_irradiance_data_to_csv(event_data, output_path):
    """
    Save the irradiance data to a CSV file.
    
    Parameters:
    event_data (list): List of dictionaries containing event data.
    output_path (str): Path to save the CSV file.
    """
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(event_data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def load_eve_scaled_irradiances(event_num):
    """
    Load the EVEScaledIrradiances.sav file for a specific event.
    
    Parameters:
    event_num (int): The event number.
    
    Returns:
    dict: A dictionary containing the scaled irradiance data, or None if the file is not found.
    """
    base_path = os.path.expanduser('~/Dropbox/Research/Woods_LASP/Analysis/Coronal Dimming Analysis/Two Two Week Period/EVEPlots/Corrected/')
    event_path = os.path.join(base_path, f'Event{event_num}', 'Warm correction', 'EVEScaledIrradiances.sav')
    
    if os.path.exists(event_path):
        data = readsav(event_path)
        
        # Extract relevant data from the SAV file
        # Adjust the keys based on the actual structure of your SAV file
        scaled_irradiance = data['correctedevedimmingcurves'][3, :]
        
        # Convert time information from Julian Date to ISO8601 format
        if 'evetimejd' in data:
            jd_times = data['evetimejd']
            time_iso = [Time(jd, format='jd').iso for jd in jd_times]
            
            # Create a dictionary with the extracted data
            scaled_data = {
                'time_iso': time_iso,
                'scaled_irradiance': scaled_irradiance
            }
            
            return scaled_data
        else:
            print(f"Required time fields not found in EVEScaledIrradiances.sav for Event{event_num}")
            return None
    else:
        print(f"File not found: {event_path}")
        return None

def find_min_scaled_irradiance(eve_scaled, preflare_time, reference_irradiance):
    """
    Find the minimum scaled irradiance within a 4-hour window after preflare_time,
    converting from percentage units back to absolute units.
    
    Parameters:
    eve_scaled (dict): Dictionary containing scaled irradiance data.
    preflare_time (str): ISO8601 timestamp of the preflare time.
    reference_irradiance (np.ndarray): Reference irradiance curve to scale back to absolute units.
    
    Returns:
    float: Minimum scaled irradiance in absolute units, or None if no data is found.
    """
    # Convert preflare_time to a datetime object
    preflare_datetime = datetime.datetime.fromisoformat(preflare_time)
    end_time = preflare_datetime + datetime.timedelta(hours=4)
    
    # Convert times in eve_scaled to datetime objects
    times_datetime = [datetime.datetime.fromisoformat(t) for t in eve_scaled['time_iso']]
    
    # Find indices within the specified time window
    indices_within_window = [i for i, t in enumerate(times_datetime) if preflare_datetime <= t <= end_time]
    
    if indices_within_window:
        # Find the reference irradiance value at the preflare time
        preflare_index = next((i for i, t in enumerate(times_datetime) 
                              if t == preflare_datetime or abs((t - preflare_datetime).total_seconds()) < 60), None)
        
        if preflare_index is not None:
            reference_value = reference_irradiance[preflare_index]
            # If the value is NaN, find the closest non-NaN value
            if np.isnan(reference_value):
                # Calculate time differences for all indices
                time_diffs = [abs((t - preflare_datetime).total_seconds()) for t in times_datetime]
                # Sort indices by time difference, excluding the preflare index
                sorted_indices = [i for i in np.argsort(time_diffs) if i != preflare_index]
                # Find the closest non-NaN value
                for idx in sorted_indices:
                    if not np.isnan(reference_irradiance[idx]):
                        reference_value = reference_irradiance[idx]
                        break
        else:
            # If no exact match, use the closest time point
            time_diffs = [abs((t - preflare_datetime).total_seconds()) for t in times_datetime]
            closest_idx = np.argmin(time_diffs)
            reference_value = reference_irradiance[closest_idx]
            # If the closest value is NaN, find the next closest non-NaN value
            if np.isnan(reference_value):
                sorted_indices = np.argsort(time_diffs)
                for idx in sorted_indices:
                    if not np.isnan(reference_irradiance[idx]):
                        reference_value = reference_irradiance[idx]
                        break
        
        # Find the minimum scaled irradiance within the window
        min_scaled_percent = np.nanmin([eve_scaled['scaled_irradiance'][i] for i in indices_within_window])
        
        # Convert from percentage back to absolute units
        min_scaled_absolute = -1 * min_scaled_percent * reference_value / 100.0
        
        return min_scaled_absolute
    else:
        return None

# Main code
preflare_times_file = os.path.expanduser('~/Dropbox/Research/Woods_LASP/Analysis/Coronal Dimming Analysis/Two Two Week Period/AngelosDaveJamesEvents_21Aug2014_YYYYDOYSOD_ManualPreFlareTimes.txt')
preflare_times = load_preflare_times(preflare_times_file)

# List to store all event data
all_event_data = []

# Loop through events and their corresponding preflare times
for event_num, preflare_time in enumerate(preflare_times, start=1):
    eve_lines = load_eve_lines(event_num)
    eve_scaled = load_eve_scaled_irradiances(event_num)
    
    if eve_lines is not None and eve_scaled is not None:
        preflare_irradiance = find_irradiance_at_nearest_time(eve_lines, preflare_time)
        min_irradiances = find_min_irradiance_within_window(eve_lines, preflare_time)
        
        reference_irradiance = eve_lines['irradiance'][:, 3] # Get the reference irradiance curve (index 3 = 17.1 nm)
        min_scaled_absolute = find_min_scaled_irradiance(eve_scaled, preflare_time, reference_irradiance)
        
        # Create a dictionary for this event
        event_dict = {
            'event_num': event_num,
            'approximate_start_time': preflare_time,
            'min_171_by_284': min_scaled_absolute
        }
        for wavelength, value in preflare_irradiance.items():
            abbreviated_wavelength = f"{wavelength:.3g}"
            event_dict[f'preflare_{abbreviated_wavelength}'] = value
        
        # Add minimum irradiance values with prefix 'min_' and abbreviated wavelength
        for wavelength, value in min_irradiances.items():
            abbreviated_wavelength = f"{wavelength:.3g}"
            event_dict[f'min_{abbreviated_wavelength}'] = value
        
        # Add to the list of all events
        all_event_data.append(event_dict)

# Save all event data to a CSV file
output_path = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-jmason86@gmail.com/.shortcut-targets-by-id/1rBkyqcRk5Ta93SG36ZmUtdMjQ8KMLtsM/ADSPS Shared/data/')
output_filename = 'two-two-week-period-irradiance-data.csv'
save_irradiance_data_to_csv(all_event_data, output_path + output_filename)
