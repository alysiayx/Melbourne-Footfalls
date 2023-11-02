import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from sklearn.metrics import r2_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
# from validclust import dunn
from autoelbow_rupakbob import autoelbow
from scipy.interpolate import CubicSpline

from basic_funs import *

def find_elbow(distortions):
  x1, y1 = 1, distortions[0]
  x2, y2 = len(distortions), distortions[-1]

  distances = []
  for i in range(len(distortions)):
    x0 = i+2
    y0 = distortions[i]
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
    distances.append(numerator/denominator)

  return distances.index(max(distances))

def transpose_data(data):
  df = data.copy()
  df = df.T
  df.index = pd.to_datetime(df.index)
  return df

def untranspose_data(data):
  df = data.copy()
  df = df.T
  return df

def find_full_data(data, aggregation=None):
  sensor_no_missing_data = {}
  
  for sensor in data.index:
    years_with_no_missing_data = []
    consecutive_years = []
    
    years = sorted(set(int(str(col)[:4]) for col in data.columns if str(col)[:4].isdigit()))
    for year in years:
      columns_for_year = [col for col in data.columns if str(year) in str(col)]
      subset = data.loc[sensor, columns_for_year]
      if not subset.isnull().values.any(): # no missing value
        if subset.shape[0] > 2:
          years_with_no_missing_data.append(year)
        consecutive_years.append(year)
      else:
        if aggregation == 'year': # check for consecutive years
          if consecutive_years and len(consecutive_years) > 2:
            if sensor in sensor_no_missing_data:
              sensor_no_missing_data[sensor].append(consecutive_years)
            else:
              sensor_no_missing_data[sensor] = [consecutive_years]
          # Reset consecutive_years list
          consecutive_years = []
          
    # Handling the last group of consecutive years
    if aggregation == 'year' and consecutive_years and len(consecutive_years) > 2:
      if sensor in sensor_no_missing_data:
        sensor_no_missing_data[sensor].append(consecutive_years)
      else:
        sensor_no_missing_data[sensor] = [consecutive_years]
    elif years_with_no_missing_data:
      sensor_no_missing_data[sensor] = years_with_no_missing_data
      
  return sensor_no_missing_data

def find_interpolation_methods(df, save_path, aggregation=None, seed=42): 
  # find the optimal method to fill missing value

  np.random.seed(seed) 
  data = df.copy()
  # ------------------- compute average missing rates -------------------
  print("compute average missing rates...")
  missing_value_proportions = data[data.isnull().any(axis=1)].isnull().mean(axis=1)
  # missing_rate = missing_value_proportions.median()
  missing_rate = missing_value_proportions.mean()

  missing_value_proportions_df = missing_value_proportions.reset_index()
  missing_value_proportions_df.columns = ['Index', 'MissingValueProportion']
  save_data(missing_value_proportions_df, save_dir=save_path, file_name='missing_rate_per_sensor.xlsx')
  
  print("average missing value rate for rows with missing values:", missing_rate)

  # ------------------- find sensors have full data -------------------
  print("find sensors have full data...")
  sensor_no_missing_data = find_full_data(data, aggregation=aggregation)
  df_sensor_no_missing_data = pd.DataFrame(list(sensor_no_missing_data.items()), columns=['Sensor', 'Years'])

  print("sensors and corresponding years with no missing data:", sensor_no_missing_data)
  if df_sensor_no_missing_data.empty:
    raise ValueError("All sensors have missing values every year.")

  save_data(df_sensor_no_missing_data, save_dir=save_path, file_name='sensor_no_missing_data.xlsx')

  # ------------------- estimate the performance of interpolation methods -------------------
  print("estimate the performance of interpolation methods...")
  methods = [
      {'method': 'time'},
      {'method': 'linear'},
      # {'method': 'cubic_spine'},
      {'method': 'quadratic', 'order': 2},
      {'method': 'cubic', 'order': 3},
      {'method': 'polynomial', 'order': 5},
      {'method': 'polynomial', 'order': 3},
      {'method': 'slinear'},
      {'method': 'zero'},
      {'method': 'nearest'}
  ]

  all_r2_scores = {method['method']: [] for method in methods}
  all_mae_scores = {method['method']: [] for method in methods}
  remove = []

  # Iterate over sensors and years with no missing data, and perform interpolation and R2 computation
  for sensor, years_groups in sensor_no_missing_data.items():
    for years in years_groups:
      if isinstance(years, int):
        years = [years]
      columns_for_years_group = [col for year in years for col in data.columns if str(year).zfill(4) in str(col)]
      data_filtered = data.loc[[sensor], columns_for_years_group]
      # print(f"sensor: {sensor}, years: {columns_for_years_group}")

      # Get the complete rows before interpolation and save them for later comparison
      original_complete_rows = data_filtered.copy()
      original_complete_rows_ = transpose_data(original_complete_rows)

      # Introduce missing_rate% missing values randomly
      random_missing_size = int(round(missing_rate, 2) * data_filtered.size)
      # print(f"the data has size {data_filtered.size} and we introduce {random_missing_size} random missing values.")

      # Exclude the first and last index when creating the list of indices to choose from
      nan_idx_pool = data_filtered.columns[1:-1]
      nan_idx = np.random.choice(nan_idx_pool, size=random_missing_size, replace=False)

      complete_rows_before = data_filtered.copy()
      complete_rows_before.loc[complete_rows_before.index[0], nan_idx] = np.nan
      complete_rows_before_ = transpose_data(complete_rows_before)
      
      # print(f"assign missing data: {complete_rows_before.isna().sum().sum()} / {data_filtered.size}")

      for method in methods:
        # Interpolate missing values
        method_name = method['method']
        order = method.get('order', None)
        try:
          # if method_name == 'cubic_spine':
          #   cs = CubicSpline(complete_rows_before_.dropna().index, complete_rows_before_.dropna())
          #   data_interpolated = pd.DataFrame(cs(complete_rows_before_.index), index=complete_rows_before_.index)
          # else:
          #   data_interpolated = complete_rows_before_.interpolate(method=method_name, order=order)
          data_interpolated = complete_rows_before_.interpolate(method=method_name, order=order)
          data_interpolated[data_interpolated < 0] = 0
        except:
          # print(f"{method} is not suitable.")
          if method not in remove:
            remove.append(method)
          continue

        # Compute scores for the row using original complete rows and interpolated data
        r2_score_for_sensor_year = r2_score(original_complete_rows_.values, data_interpolated.values)
        mae_for_sensor_year = mean_absolute_error(original_complete_rows_.values, data_interpolated.values)

        all_r2_scores[method_name].append(r2_score_for_sensor_year)
        all_mae_scores[method_name].append(mae_for_sensor_year)

  results = []

  for method in methods:
    if method not in remove:
      method_name = method['method']
      order = method.get('order', None)
      r2_scores = all_r2_scores[method_name]
      mae_scores = all_mae_scores[method_name]
      results.append({
        'Method': f"{method_name} (order {order})" if order else method_name,
        'R2': np.mean(r2_scores),
        'MAE':np.mean(mae_scores),
        'Method Details': method
      })

  results_df = pd.DataFrame(results)

  results_df.sort_values(by=["R2", "MAE"], ascending=[False, True], inplace=True)

  # results_df['R2 Rank'] = results_df['R2'].rank(ascending=False)
  # results_df['MAE Rank'] = results_df['MAE'].rank(ascending=True)
  # results_df['Average Rank'] = results_df[['R2 Rank', 'MAE Rank']].mean(axis=1)
  # results_df.sort_values(by=["Average Rank"], ascending=True, inplace=True)

  print_table(results_df)
  print(f"The best method is: {results_df.iloc[0][0]} with an average R2 score of {results_df.iloc[0][1]:.2f} and an average MAE score of {results_df.iloc[0][2]:.2f}")
  if pd.isna(results_df.iloc[0][1]):
    raise ValueError("R2 score shouldn't be nan.")

  save_data(results_df, save_dir=save_path, file_name='best_interpolation_methods.xlsx')
  return results_df['Method Details'].iloc[0], results_df

def fill_missinng_data(data, save_path, aggregation=None, seed=42):
  # the data should be pivoted
  print(f"find optimal interpolation methods and fill the missing data with size {data.shape}......")
  np.random.seed(seed) 
  df = data.copy()
  best_interpolation, _ = find_interpolation_methods(df, save_path, aggregation, seed)
  missing_before = df.isna().sum().sum()
  
  df = transpose_data(df)

  # interpolate the internal gaps
  # the interpolation will be applied row-wise across index (Date_Time)
  if 'order' in best_interpolation.keys():
    df.interpolate(method=best_interpolation['method'], order=best_interpolation['order'], inplace=True)
  else:
    df.interpolate(method=best_interpolation['method'], inplace=True)
  
  df[df < 0] = 0
  
  missing_after = df.isna().sum().sum()
  
  # fill over gaps
  df.fillna(value=0, inplace=True)

  df = untranspose_data(df)

  missing_after_zero = df.isna().sum().sum()
  print(f"number of missing values before interpolation: {missing_before}")
  print(f"number of missing values after interpolation with best methods: {missing_after}")
  print(f"number of missing values after interpolation with 0: {missing_after_zero}")

  return df

def scale_data(data):
  df = data.copy()

  # standardizes data along columns (features)
  scaler = StandardScaler()

  if df.index.name == 'Date_Time': # the features (Date_Time) should be columns
    df = df.T
  df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
  return df_scaled

def find_optimal_k(df, save_path, algorithm='kmeans', metric='euclidean', n_init=10, max_iter=300, seed=42, verbose=True):
  print(f"now is finding the optimal k for {algorithm}")
  data = df.copy()
  distortions = [] # the sum of squared distances of samples to their nearest cluster center
  ss_scores = []
  db_scores = []
  ch_scores = []
  di_scores = []

  configs = {
    "n_init": n_init,
    "max_iter": max_iter,
    "verbose": verbose,
    "random_state": seed
  }
  
  for i in range(2, 16):
    if algorithm == 'kmeans':
      model = TimeSeriesKMeans(n_clusters=i, metric=metric, init='k-means++', **configs)
    elif algorithm == 'kshape':
      model = KShape(n_clusters=i, **configs)
    elif algorithm == 'kernelkmeans':
      model = KernelKMeans(n_clusters=i, kernel="gak", kernel_params={"sigma": "auto"}, **configs)
      
    model.fit(data)
    
    distortions.append(model.inertia_)
    ss_scores.append(silhouette_score(data, model.labels_))
    db_scores.append(davies_bouldin_score(data, model.labels_))
    ch_scores.append(calinski_harabasz_score(data, model.labels_))
    # dist = pairwise_distances(data)
    # di_scores.append(dunn(dist, model.labels_))
  
  df_scores = pd.DataFrame({
    'Number_of_Clusters': range(2, 16),
    'Distortion': distortions,
    'Silhouette_Score': ss_scores,
    'Davies_Bouldin': db_scores,
    'Calinski_Harabasz': ch_scores,
    # 'Dunn': di_scores
  })

  best_silhouette = df_scores.sort_values(by='Silhouette_Score', ascending=False, inplace=False).iloc[0] # higher is better
  best_ch = df_scores.sort_values(by='Calinski_Harabasz', ascending=False, inplace=False).iloc[0] # higher is better
  best_db = df_scores.sort_values(by='Davies_Bouldin', inplace=False).iloc[0] # lower is better
  elbow_point = df_scores['Number_of_Clusters'].iloc[find_elbow(df_scores['Distortion'].values.tolist())]
  print_table(df_scores)

  print("Best k based on Silhouette Score:", best_silhouette['Number_of_Clusters'])
  print("Best k based on Elbow Point:", elbow_point)
  # print("Best k based on Elbow Point (autoelbow):", autoelbow.auto_elbow_search(data))
  print("Best k based on Davies-Bouldin Index:", best_db['Number_of_Clusters'])
  print("Best k based on Calinski-Harabasz Index:", best_ch['Number_of_Clusters'])

  # Ranking each score appropriately
  df_scores['Silhouette_Rank'] = df_scores['Silhouette_Score'].rank(ascending=False)
  df_scores['Davies_Bouldin_Rank'] = df_scores['Davies_Bouldin'].rank()
  df_scores['Calinski_Harabasz_Rank'] = df_scores['Calinski_Harabasz'].rank(ascending=False)

  # Compute the aggregate rank by summing individual ranks
  df_scores['Aggregate_Rank'] = df_scores['Silhouette_Rank'] + df_scores['Davies_Bouldin_Rank'] + df_scores['Calinski_Harabasz_Rank']

  # Sort the dataframe based on Aggregate Rank to find the best k
  df_scores = df_scores.sort_values(by='Aggregate_Rank')
  best_k = sorted(list(set([int(df_scores.iloc[0]['Number_of_Clusters']), elbow_point])))

  print("Best k based on the aggregate ranking of all scores:", best_k)
  save_data(df_scores, save_dir=save_path, file_name='best_k.xlsx')
  return best_k, df_scores

if __name__ =='__main__':
  from pathlib import Path
  data = pd.read_csv('footfall_merged.csv') # the data should be unpivoted
  data = data[['New_Sensor_Name', 'Date_Time', 'Hourly_Counts']]
  data['Date_Time'] = pd.to_datetime(data['Date_Time'])
  data.set_index('Date_Time', inplace=True)
  data = data.groupby('New_Sensor_Name').resample('Y').mean().reset_index()
  data = data.pivot(index='New_Sensor_Name', columns='Date_Time', values='Hourly_Counts')
  fill_missinng_data(data, Path('./'), aggregation=None)

