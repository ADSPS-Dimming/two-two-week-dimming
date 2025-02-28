import os
from scipy.io import readsav
import datetime
import pandas as pd  # Import pandas
import numpy as np  # Import numpy for handling arrays

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

# Main code
preflare_times_file = os.path.expanduser('~/Dropbox/Research/Woods_LASP/Analysis/Coronal Dimming Analysis/Two Two Week Period/AngelosDaveJamesEvents_21Aug2014_YYYYDOYSOD_ManualPreFlareTimes.txt')
preflare_times = load_preflare_times(preflare_times_file)

# List to store all event data
all_event_data = []

# Loop through events and their corresponding preflare times
for event_num, preflare_time in enumerate(preflare_times, start=1):
    eve_lines = load_eve_lines(event_num)
    if eve_lines is not None:
        preflare_irradiance = find_irradiance_at_nearest_time(eve_lines, preflare_time)
        min_irradiances = find_min_irradiance_within_window(eve_lines, preflare_time)
        
        # Create a dictionary for this event
        event_dict = {
            'event_num': event_num,
            'approximate_start_time': preflare_time
        }
        
        # Add preflare irradiance values with prefix 'preflare_' and abbreviated wavelength
        for wavelength, value in preflare_irradiance.items():
            # Format wavelength to 3 digits
            abbreviated_wavelength = f"{wavelength:.3g}"
            event_dict[f'preflare_{abbreviated_wavelength}'] = value
        
        # Add minimum irradiance values with prefix 'min_' and abbreviated wavelength
        for wavelength, value in min_irradiances.items():
            # Format wavelength to 3 digits
            abbreviated_wavelength = f"{wavelength:.3g}"
            event_dict[f'min_{abbreviated_wavelength}'] = value
        
        # Add to the list of all events
        all_event_data.append(event_dict)

# Save all event data to a CSV file
output_path = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-jmason86@gmail.com/.shortcut-targets-by-id/1rBkyqcRk5Ta93SG36ZmUtdMjQ8KMLtsM/ADSPS Shared/data/')
output_filename = 'two-two-week-period-irradiance-data.csv'
save_irradiance_data_to_csv(all_event_data, output_path + output_filename)
