import osmnx as ox
import pandas as pd
import numpy as np
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt

from basic_funs import *

def haversine(lon1, lat1, lon2, lat2):
  """
  Calculate the great-circle distance between two points 
  on the Earth (specified in decimal degrees)
  """
  # Convert decimal degrees to radians
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

  # Haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a))
  r = 6371  # Radius of Earth in kilometers. Use 3956 for miles
  return c * r

def fetch_amenities_count(lat, lon, distance, amenity_types):
  amenity_count = {}
  
  for amenity in amenity_types:
    # print(f"Fetching {amenity}...")
    try:
      gdf = ox.features_from_point((lat, lon), tags={'amenity': amenity}, dist=distance)
      if not gdf.empty:
        amenity_count[amenity] = len(gdf)
    except:
      pass
      
  return amenity_count


def sensor_amenities_count(sensor_df, distance, amenity_types):
  amenities_counts = []
  
  for index, row in sensor_df.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    count = fetch_amenities_count(lat, lon, distance, amenity_types)
    amenities_counts.append(count)
      
  amenities_count_df = pd.DataFrame(amenities_counts, index=sensor_df.index)
  return amenities_count_df

def fetch_amenities(lat, lon, distance, amenity_types):  
  gdfs = []
  for amenity in amenity_types:
    # print(f"Fetching {amenity}...")
    try:
      gdf = ox.features_from_point((lat, lon), tags={'amenity': amenity}, dist=distance)
      if not gdf.empty:
        gdf.reset_index(inplace=True)
        gdf = gdf[['osmid', 'amenity', 'geometry']]
        gdfs.append(gdf)
    except Exception as e:
      # print(f"Error fetching {amenity}: {e}")
      pass
  
  combined_gdf = pd.concat(gdfs, ignore_index=True) if gdfs else pd.DataFrame()

  return combined_gdf

def find_amenity_types_in_city(place_name, save_amenities):

  # Fetch all geometries in the area
  gdf = ox.geometries_from_place(place_name, tags={'amenity': True})

  # Filter to keep only the 'amenity' tag
  amenities = gdf[gdf['amenity'].notnull()]

  amenity_types = amenities['amenity'].unique().tolist()

  if not save_amenities.exists():
    with open(save_amenities, 'w') as f:
      for amenity in amenity_types:
        f.write(f"{amenity}\n")

    print("Amenity types saved to amenity_types_melbourne.txt")
  else:
    amenity_types = pd.read_csv(save_amenities, header=None)[0].tolist()
  
  return amenity_types

def fetch_amenities_per_sensor(sensor, distance_list, valid_amenity_type, 
                              save_path_total, save_path_per_sensor):
    all_gdfs = []
    sensor_gdfs = defaultdict(dict)

    for distance in distance_list:
        print(distance)
        for _, row in sensor.iterrows():
            lat, lon = row['Latitude'], row['Longitude']
            gdfs = fetch_amenities(lat, lon, distance, valid_amenity_type)
            
            if not gdfs.empty:
                all_gdfs.append(gdfs)
                sensor_gdfs[row['Sensor_Name']][distance] = gdfs['osmid'].tolist()  # save all amenities
        
        valid_amenities_df = pd.concat(all_gdfs, ignore_index=True) if all_gdfs else pd.DataFrame()
        valid_amenities_df = valid_amenities_df.drop_duplicates()
        print(f'{valid_amenities_df.shape[0]} amenities have been found')
        save_data(valid_amenities_df, Path(str(save_path_total) + str(distance) + '.csv'))

        # save nearby amenties found for each sensor
        data_for_df = []
        for sensor_name, distances in sensor_gdfs.items():
            for dist, osmids in distances.items():
                data_for_df.append({
                    'Sensor_Name': sensor_name,
                    'Distance': dist,
                    'Osmids': osmids
                })

        sensor_gdfs_df = pd.DataFrame(data_for_df)

        # Check if save_path_per_sensor exists and append if it does
        save_path = Path(save_path_per_sensor)
        if save_path.exists():
            existing_df = pd.read_csv(save_path)
            sensor_gdfs_df = pd.concat([existing_df, sensor_gdfs_df], ignore_index=True)
        
        save_data(sensor_gdfs_df, save_path)

    return sensor_gdfs_df

def fetch_total_amenities_in_circle(sensor, valid_amenities, distance_list, save_path, rewrite=False):
  latitudes = sensor['Latitude'].values
  longitudes = sensor['Longitude'].values
  center_lat = np.mean(latitudes)
  center_lon = np.mean(longitudes)

  max_dist = max([haversine(lon, lat, center_lon, center_lat) for lat, lon in zip(latitudes, longitudes)])
  print(max_dist)

  for distance in distance_list:
    save_path_dist = Path(str(save_path) + str(distance) + '.csv')
    if not save_path_dist.exists() and rewrite==False:
      print(distance)
      extended_radius = max_dist * 1000 + distance 

      amenities_circle = fetch_amenities(center_lat, center_lon, extended_radius, valid_amenities)

      amenities_circle = amenities_circle.drop_duplicates()
      save_data(amenities_circle, save_path_dist)
