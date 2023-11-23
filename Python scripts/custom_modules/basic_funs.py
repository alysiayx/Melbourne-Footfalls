# -*- coding: utf-8 -*-
# @Author: alysia
# @Date:   2023-09-16 22:51:37
# @Last Modified by:   alysia
# @Last Modified time: 2023-09-17 03:40:38

import re
import zipfile
import gzip
import numpy as np
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from tabulate import tabulate

class Config:
  # convert the dictionary to an object
  def __init__(self, **entries):
    # class takes a dictionary as input 
    # updates the class's __dict__ attribute with the entries from the dictionary.
    # In Python, every object has an attribute named __dict__
    self.__dict__.update(entries)

def split_and_save_data(data, save_dir, save_subdir, rewrite=False, return_data=False):
  # Define the boundaries
  pre_covid_end = '2020-02-29'
  during_covid_start = '2020-03-01'
  during_covid_end = '2021-10-21'

  df = data.copy()

  # Convert to datetime if not already
  df['Date_Time'] = pd.to_datetime(df['Date_Time'])

  # Split the data
  pre_covid_data = df[df['Date_Time'] <= pre_covid_end]
  during_covid_data = df[(df['Date_Time'] >= during_covid_start) & (df['Date_Time'] <= during_covid_end)]
  post_covid_data = df[df['Date_Time'] > during_covid_end]

  save_path = Path(save_dir) / save_subdir
  save_path.mkdir(parents=True, exist_ok=True)

  if rewrite == True or (save_path / 'footfall_merged_post_cov.csv').exists() == False:
    df.to_csv(save_path / 'footfall_merged.csv', index=False, header=True)
    pre_covid_data.to_csv(save_path / 'footfall_merged_pre_cov.csv', index=False, header=True)
    during_covid_data.to_csv(save_path / 'footfall_merged_during_cov.csv', index=False, header=True)
    post_covid_data.to_csv(save_path / 'footfall_merged_post_cov.csv', index=False, header=True)

  if return_data == True:
    return pre_covid_data, during_covid_data, post_covid_data

def check_duplicates(data, columns='all', print_msg='data', keep=False):
  if columns == 'all':
    df = data.copy()
  else:
    df = data[columns].copy()
    print_msg = f'{print_msg} {columns}'
  
  duplicates = data[df.duplicated(keep=keep)]
  print(f"Duplicates in {print_msg}: {len(duplicates)}")
  
  if not duplicates.empty:
    print(tabulate(duplicates, headers=duplicates.columns, tablefmt="outline"))
  
  return duplicates

def set_path(save_dir, save_subdir=None, file_name=None):
  if save_subdir is not None:
    save_path = save_dir / save_subdir
  else:
    save_path = save_dir
  save_path.mkdir(parents=True, exist_ok=True)
  print(f"{save_path} created.")

  if file_name is not None:
    return save_path / file_name
  else:
    return save_path

def save_data(data, save_dir, save_subdir=None, file_name=None, index=False, header=True, 
              index_label=None, rewrite=True):
  save_path = set_path(save_dir, save_subdir, file_name)

  if rewrite == True or not save_path.exists():
    if save_path.exists():
      print(f"{save_path} will be updated.")
    else:
      print(f"{save_path} will be saved.")
    
    # Check if data has MultiIndex columns and is being written to an Excel file
    if isinstance(data.columns, pd.MultiIndex) and file_name.endswith('.xlsx'):
      if not index:  # If index is set to False
        print("Data has MultiIndex columns. Resetting columns and setting 'index=True'.")
        data.columns = [' '.join(col).strip() for col in data.columns.values]
        index = True

    if file_name.endswith('.xlsx'):
      data.to_excel(save_path, index=index, header=header, index_label=index_label)
    else:
      data.to_csv(save_path, index=index, header=header, index_label=index_label)
    
    print(f"{save_path} saved.")
  else:
    if save_path.exists():
      print(f"{save_path} exists but not be updated.")

def standardize_string(s):
  return re.sub(r'\s*-\s*', '-', s)

def process_sensor_loc_data(sensor_locations):
  df = sensor_locations.copy()
  df['Sensor_Description'] = df['Sensor_Description'].apply(standardize_string)

  def custom_format(row):
    combined = sorted(zip(row['Location_ID'], row['Sensor_Name']))
    unique_names = []
    for _, name in combined:
      if name not in unique_names:
        unique_names.append(name)
    sensor_name_str = ", ".join(unique_names)
    location_id_str = ", ".join(map(str, [loc for loc, _ in combined]))
    return f"{row['Sensor_Description']} | {sensor_name_str} [{location_id_str}]"

  grouped = df.groupby(['Sensor_Description', 'Location']).agg({
    'Sensor_Name': lambda x: list(x),
    'Location_ID': lambda x: list(x),
  }).reset_index()

  grouped['New_Sensor_Name'] = grouped.apply(custom_format, axis=1)

  exploded_grouped = grouped.explode('Location_ID')
  merged_df = pd.merge(df, exploded_grouped[['Location_ID', 'New_Sensor_Name']], on='Location_ID', how='left')

  cols = list(df.columns)
  cols.insert(cols.index('Sensor_Name') + 1, 'New_Sensor_Name')
  merged_df = merged_df[cols]

  print(f"The initial number of locations is {len(df['Location_ID'].unique())}")
  print(f"The number of unique New_Sensor_Names after merging is {len(merged_df['New_Sensor_Name'].unique())}")

  return merged_df

def has_timezone_indicator(s):
  s = s.astype(str)
  return s.str.contains(r'[+\-]\d{2}:\d{2}', regex=True).any()

def format_datetime_remove_duplicates(hist_data, curr_data):
  hist_df, curr_df = hist_data.copy(), curr_data.copy()

  # format datetime column
  hist_df['Date_Time'] = pd.to_datetime(hist_df['Date_Time'], format='%B %d, %Y %I:%M:%S %p')

  # the data collected after 2022 using ISO 8601 date and time format, and the timezone is UTC
  curr_df['SensingDateTime(Hour)'] = pd.to_datetime(curr_df['SensingDateTime(Hour)'], errors='coerce')

  # check if there are any NaT (Not-a-Time) values, which means there were strings that couldn't be converted to datetimes
  if curr_df['SensingDateTime(Hour)'].isna().any():
    print("There are invalid datetime strings!")
    print(curr_df[curr_df['SensingDateTime(Hour)'].isna()])

  # remove the timezone information
  try:
    curr_df['SensingDateTime(Hour)'] = curr_df['SensingDateTime(Hour)'].dt.tz_localize(None)
  except:
    curr_df['SensingDateTime(Hour)'].apply(lambda x: x.replace(tzinfo=None))
  
  # Check if any timestamp contains characters associated with timezones
  if has_timezone_indicator(curr_df['SensingDateTime(Hour)']):
    # Remove timezone manually
    print("manually remove the timezone")
    curr_df['SensingDateTime(Hour)'] = curr_df['SensingDateTime(Hour)'].astype(str)
    curr_df['SensingDateTime(Hour)'] = curr_df['SensingDateTime(Hour)'].str.slice(0, -6)
    if not has_timezone_indicator(curr_df['SensingDateTime(Hour)']):
      print("Timezone information has been successfully removed!")
    else:
      print("Failed to remove all timezone information!")
  else:
    print("No timezone information found in the timestamps!")

  # remove duplicate rows
  num_duplicates_hist = hist_df.duplicated().sum()
  num_duplicates_curr = curr_df.duplicated().sum()
  hist_df.drop_duplicates(inplace=True)
  print(f"Number of duplicate rows in historical data: {num_duplicates_hist}")
  print(f"Number of duplicate rows in current data: {num_duplicates_curr}")
  return hist_df, curr_df

def match_sensor_details(data1, data2, column_pair1, column_pair2):
  # Find if column_pair1 in data1 is equivalent to column_pair2 in data2

  df1, df2 = data1.copy(), data2.copy()

  df1['Sensor_Pair'] = df1[column_pair1[0]].astype(str) + "_" + df1[column_pair1[1]].astype(str)
  df2['Sensor_Pair'] = df2[column_pair2[0]].astype(str) + "_" + df2[column_pair2[1]].astype(str)
  common_pairs = df1[df1['Sensor_Pair'].isin(df2['Sensor_Pair'])]

  similarity_ratio = len(common_pairs) / len(df1)

  print(f"The similarity is {similarity_ratio*100:.2f}%")

  if similarity_ratio >= 0.8:
    print(f"The pair of {column_pair1} in data1 may equivalent to {column_pair2} in data2.")
  else:
    print("They are NOT equivalent")
  
  return similarity_ratio

def unify_sensor_name(data1, data2, column_pair1, column_pair2):
  df1, df2 = data1.copy(), data2.copy()

  # sort the given lists of column names so that columns ending in "_ID" come first
  column_pair1 = sorted(column_pair1, key=lambda x: 0 if x.endswith('_ID') else 1)
  column_pair2 = sorted(column_pair2, key=lambda x: 0 if x.endswith('_ID') else 1)

  merge_df = pd.merge(df1[column_pair1], df2[column_pair2], left_on=column_pair1[0], right_on=column_pair2[0])
  uncommon_pairs = merge_df[merge_df[column_pair1[1]] != merge_df[column_pair2[1]]]
  unique_uncommon_pairs = uncommon_pairs.drop_duplicates()

  # iterate through the unique uncommon pairs
  for _, row in unique_uncommon_pairs.iterrows():
    # update the Sensor_Name in data1
    df1.loc[df1[column_pair1[0]] == row[column_pair1[0]], column_pair1[1]] = row[column_pair2[1]]

  return df1, unique_uncommon_pairs

def merge_hist_curr_sensor_data(hist_data, curr_data, sensor_data):
  hist_df, curr_df, sensor_df = hist_data.copy(), curr_data.copy(), sensor_data.copy()

  # find if (Sensor_Name, Sensor_ID) in footfall_counts_09_22 is equivalent to 
  # (Sensor_Description, Location_ID) in sensor_locations
  column_pair1 = ['Sensor_Name', 'Sensor_ID']
  column_pair2 = ['Sensor_Description', 'Location_ID']
  print(f"1. Find if {column_pair1} in footfall_counts_09_22 is equivalent to {column_pair2} in sensor_locations")
  similarity1 = match_sensor_details(hist_df, sensor_df, column_pair1, column_pair2)

  # find if (Sensor_ID, Sensor_Name) in footfall_counts_09_22 is equivalent to 
  # (Location_ID, Sensor_Name) in sensor_locations
  column_pair3 = ['Sensor_ID', 'Sensor_Name']
  column_pair4 = ['Location_ID', 'Sensor_Name']
  print(f"2. Find if {column_pair3} in footfall_counts_09_22 is equivalent to {column_pair4} in sensor_locations")
  similarity2 = match_sensor_details(hist_df, sensor_df, column_pair3, column_pair4)

  # unify sensor name in historical data
  if similarity1 > similarity2:
    print(f"{column_pair1} in footfall_counts_09_22 is equivalent to {column_pair2} in sensor_locations")
    hist_df, uncommon_pairs = unify_sensor_name(hist_df, sensor_df, column_pair1, column_pair2)
  else:
    print(f"{column_pair3} in footfall_counts_09_22 is equivalent to {column_pair4} in sensor_locations")
    hist_df, uncommon_pairs = unify_sensor_name(hist_df, sensor_df, column_pair3, column_pair4)
  
  # merge data sets
  hist_df_subset = hist_df[['Date_Time', 'Sensor_ID', 'Sensor_Name', 'Hourly_Counts']].copy()
  hist_df_subset.rename(columns={'Sensor_ID': 'Location_ID'}, inplace=True)

  curr_df_subset = curr_df[['SensingDateTime(Hour)', 'LocationID', 'Total_of_Directions']].copy()
  curr_df_subset.rename(columns={'LocationID': 'Location_ID', 'SensingDateTime(Hour)': 'Date_Time', 'Total_of_Directions': 'Hourly_Counts'}, inplace=True)

  # Select specific columns from sensor_locations
  sensor_locations_subset = sensor_data[['Location_ID', 'Sensor_Description', 'Installation_Date', 'Location_Type', 'Status', 'Latitude', 'Longitude', 'Location']].copy()
  sensor_locations_subset.rename(columns={'Sensor_Description': 'Sensor_Name'}, inplace=True)

  # Merge footfall_09_22_subset with sensor_locations_subset on Location_ID and Sensor_Name
  merged_df_1 = pd.merge(hist_df_subset, sensor_locations_subset, on=['Location_ID', 'Sensor_Name'], how='inner')

  # Merge footfall_23_today_subset with sensor_locations_subset on Location_ID
  merged_df_2 = pd.merge(curr_df_subset, sensor_locations_subset, on='Location_ID', how='inner')

  # Merge merged_df_1 with footfall_23_today_subset on Location_ID
  final_merged_df = pd.concat([merged_df_1, merged_df_2], ignore_index=True)

  final_merged_df = final_merged_df.drop_duplicates()

  final_merged_df['Date_Time'] = pd.to_datetime(final_merged_df['Date_Time'])

  print(f"The shape of final_merged_df before grouping {final_merged_df.shape[0]}")

  # add New_Sensor_Name to the final_merged_df
  final_merged_df = pd.merge(final_merged_df, sensor_data[['Location_ID', 'New_Sensor_Name']], on='Location_ID', how='inner')

  # Group hourly data
  final_merged_df = final_merged_df.groupby(['Date_Time', 'New_Sensor_Name']).agg({
      'Hourly_Counts': 'sum',
      'Location_ID': 'first',
      'Installation_Date': 'first',
      'Location_Type': 'first',
      'Status': 'first',
      'Latitude': 'first',
      'Longitude': 'first',
      'Location': 'first'
  }).reset_index()

  print(f"The shape of final_merged_df after grouping {final_merged_df.shape[0]}")

  # Extract Year, Month, Date, and Day of Week
  final_merged_df['Year'] = final_merged_df['Date_Time'].dt.year
  final_merged_df['Month'] = final_merged_df['Date_Time'].dt.month
  final_merged_df['MDate'] = final_merged_df['Date_Time'].dt.day
  final_merged_df['Day'] = final_merged_df['Date_Time'].dt.day_name()

  return final_merged_df, uncommon_pairs
  
def print_table(data):
  print(tabulate(data, headers=data.columns, tablefmt="outline"))

def read_file_with_stem(directory_path, file_name_to_search):
  # handle some large files may be zipped and cannot read directly

  found_files = []

  for file in directory_path.iterdir():
    if file.name.startswith(file_name_to_search + "."):
      found_files.append(file)
  
  print(f"find matched file(s): {found_files}")

  if len(found_files) >= 1:
    file = found_files[0]
    if file.suffix == '.zip': 
      # cannot directly read 'footfall_merged.csv.zip' due to "Multiple files error"
      print(f"extract file(s) in {file}")
      with zipfile.ZipFile(file, 'r') as z:
        # Extract all files inside the zip file
        z.extractall(directory_path)
        for extracted_file in directory_path.glob(file.stem + '*'):
          if extracted_file.suffix == '.csv':
            print(f"read {extracted_file}")
            data = pd.read_csv(extracted_file)
            break
    elif file.suffix in ['.csv', '.gz']:
      print(f"read {file}")
      data = pd.read_csv(file)
    # # or using gzip to process file with .gz extension
    # elif file.suffix == '.gz':
    #   with gzip.open(file, 'rt') as f:
    #     data = pd.read_csv(f)
  else:
    print(f"{len(found_files)} file(s) found for '{file_name_to_search}'. Please check.")
