import os
import pickle
import tsfel
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Dict
from collections import Counter
from pathlib import PosixPath
from sklearn.cluster import Birch
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from sklearn.metrics import silhouette_score as silhouette_score_skl
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA, IncrementalPCA

from tsfresh import extract_features

# import autocluster
# from autocluster import AutoCluster, get_evaluator, MetafeatureMapper

from modelling_funs import fill_missinng_data, find_optimal_k, find_elbow, transpose_data
from plot_funs import plot_best_k, plot_map, plot_time_series_data_sensor, plot_pca_heatmap
from basic_funs import save_data, set_path, print_table
from default_model_configs import get_default_model_configs

class TSClustering:
  def __init__(self, data: pd.DataFrame, sensor_locations: pd.DataFrame, **configs):
    """
    Parameters:
    - data: by default is unpivot (wide format) hourly footfall data
    - metric: 
      "euclidean", "dtw", "softdtw" or None
    - scale: None or
      "day", 'week', 'month', 'year', 'hour'
      'early_morning', 'morning', 'midday', 'afternoon', 'evening'
      'workday', 'weekend'
    - model: 
      "kmeans", "kshape", "kernelkmeans", "birch", "ensemble"
    - time_span: float, int or list
      "normal" (before 2020), 
      2019 (or other single year), 
      [start_date, end_date] or None
    - normalise: 
      "meanvariance", "minmax" or None
    - feature_extraction: 
      True, False or None
    - dim_reduction: 
      'PCA', 'IPCA' or None
    - "order_of_impute_agg": 
      "impute_agg_norm", "impute_norm_agg", "agg_impute_norm", or "agg_norm_impute"
    """
    self.raw_data = data.copy() # original row data
    self.data = data.copy() # data during pre-processed
    self.data_raw = None # store the data before normalisation
    self.data_raw_norm = None # store the data after normalisation
    self.sensor_locations = sensor_locations
    self.root_save_dir = configs.get('save_dir', './')

    self._initialize_configs(configs)
    self._validate_config()

  def _initialize_configs(self, configs: Dict):
    default_config = {
      "target_column": None,
      "time_column": None,
      "value_column": None,
      "time_span": None,
      "order_of_impute_agg_norm": "impute_agg_norm",
      "scale": None,
      "algorithm": "kmeans",
      "normalise": "meanvariance",
      "model_configs": None,
      "best_k": None,
      "df_scores": None,
      "df_results": None,
      "feature_extraction": None,
      "dim_reduction": None,
      "save_dir": "./",
      "seed": 42,
      "rewrite": True,
      "verbose": True,
    }
    
    # Update the default config with provided configs
    default_config.update(configs)
    self.global_configs = default_config
    
    # Set each config as an attribute
    for key, value in default_config.items():
      setattr(self, key, value)
    
    if self.model_configs is None:
      print("the model's configs are set as default values.")
      self.model_configs = get_default_model_configs(self.algorithm, self.seed)

  def _validate_config(self):
    if (self.target_column is None or 
        self.time_column is None or 
        self.value_column is None):
      raise ValueError("target_column, time_column, and value_column must be provided.")    

  def save_configs(self):
    def handle_posix_path(obj):
      if isinstance(obj, PosixPath):
        return str(obj)
      raise TypeError("Type not serializable")

    with open(self.save_dir / '_global_configs.json', 'w') as file:
      json.dump(self.global_configs, file, default=handle_posix_path)
      print('global configs saved.')
    
    print(self.model_configs)
    with open(self.save_dir / '_model_configs.json', 'w') as file:
      json.dump(self.model_configs, file)
      print('model configs saved.')
    
  def set_save_dir(self, save_dir=None):
    if save_dir is None:
      save_dir = self.root_save_dir
    
    if self.algorithm in ['kmeans', 'kshape']:
      model_name = f"{self.model_configs['metric']}_{self.algorithm}"
    elif self.algorithm == 'kernelkmeans':
      model_name = f"{self.model_configs['kernel']}_{self.algorithm}"
    elif self.algorithm in ['birch']:
      model_name = self.algorithm
      
    path = f"{model_name}_norm-{self.normalise}_scale-{self.scale}_span-{self.time_span}/" \
        f"order-{self.order_of_impute_agg_norm}_fea-{self.feature_extraction}_dr-{self.dim_reduction}"
    
    save_dir = save_dir / path
    self.save_dir = set_path(save_dir)
  
  def select_time_span(self, data=None, keep_index=None):
    print("-"*50)

    if data is None:
      data = self.data.copy()
        
    # Convert column names to datetime objects, assume the data is pivoted
    datetime_columns = pd.to_datetime(data.columns)
    
    if self.time_span is not None:
      print(f"the data shape before cutting is {data.shape}")
      
      # Mask to determine which columns to keep
      mask = np.array([False] * len(datetime_columns))
      
      if self.time_span == 'normal':
        print("select data before 2020.....")
        end_year = 2019
        mask |= (datetime_columns.year <= end_year)
      elif isinstance(self.time_span, list) and len(self.time_span) == 2: # start_year and end_year
        start_date = self.time_span[0]
        end_date = self.time_span[1]
        print(f"select data from {start_date} to {end_date}")
        mask |= (datetime_columns.year >= start_date) & (datetime_columns.year <= end_date)
      elif isinstance(self.time_span, int):
        print(f"select data in {self.time_span}")
        mask |= (datetime_columns.year == self.time_span)
      else:
        raise ValueError("Please enter correct time_span")
      
      if keep_index is None:
        data = data.loc[:, mask]
      else:
        data = data.loc[keep_index, mask]
      
      data = data.dropna(axis=0, how='all')
      data.columns.name = 'Date_Time'

      print(f"the data shape after cutting is {data.shape}")
      print(f'the data range: {data.columns[0]} - {data.columns[-1]}')
      save_data(data, save_dir=self.save_dir, file_name='data.csv', rewrite=self.rewrite, index=True)
    else:
      print("use all data.....")
    
    self.data = data

    return data

  def aggregation(self, data=None): 
    # the data is pivoted after this step
    if data is None:
      data = self.data.copy()

    print("-"*50)
    print(f"the data size before aggregation is {data.shape}")
    if self.scale is not None:
      print(f"the data will be aggregated by {self.scale}")
      df_transposed = transpose_data(data)
      if self.scale == 'year':
        data = df_transposed.resample('Y').sum().reset_index()
      elif self.scale == 'month':
        data = df_transposed.resample('M').sum().reset_index()
      elif self.scale == 'week':
        data = df_transposed.resample('W').sum().reset_index()
      elif self.scale == 'day':
        data = df_transposed.resample('D').sum().reset_index()
      elif self.scale == 'hour':
        data = df_transposed.resample('H').sum().reset_index()
      elif self.scale == 'early_morning':
        early_morning_data = df_transposed.between_time('00:00', '06:00')
        data = early_morning_data.resample('D').sum().reset_index()
      elif self.scale == 'morning':
        morning_data = df_transposed.between_time('06:00', '12:00')
        data = morning_data.resample('D').sum().reset_index()
      elif self.scale == 'midday':
        midday_data = df_transposed.between_time('12:00', '13:00')
        data = midday_data.resample('D').sum().reset_index()
      elif self.scale == 'afternoon':
        afternoon_data = df_transposed.between_time('13:00', '18:00')
        data = afternoon_data.resample('D').sum().reset_index()
      elif self.scale == 'evening':
        evening_data = df_transposed.between_time('18:00', '00:00')
        data = evening_data.resample('D').sum().reset_index()
      elif self.scale == 'weekend':
        weekend_data = df_transposed[df_transposed.index.dayofweek >= 5]
        data = weekend_data.resample('D').sum().reset_index()
      elif self.scale == 'workday':
        weekend_data = df_transposed[df_transposed.index.dayofweek < 5]
        data = weekend_data.resample('D').sum().reset_index()

      data = data.transpose()
      data.columns = data.loc[self.time_column]
      data = data.drop(self.time_column)
      data = data.reset_index().rename(columns={"index": self.target_column}).set_index(self.target_column)
      data = data.dropna(axis=1, how='all')

      are_columns_ascsending = (list(pd.to_datetime(data.columns)) == sorted(pd.to_datetime(data.columns)))
      if are_columns_ascsending == False:
        data.sort_values(axis=1, inplace=True)
      print(f"the aggregated data size is {data.shape}")
      save_data(data, save_dir=self.save_dir, file_name='aggregated_data.xlsx', index=True, rewrite=self.rewrite)
    else:
      data = data.pivot(index=self.target_column, columns=self.time_column, values=self.value_column)
      print(f"the pivoted data size is {data.shape}")
    
    self.data = data
    return data

  def impute_data(self, data=None):
    # the data should be pivoted
    if data is None:
      data = self.data.copy()
    
    print("-"*50)
    print("impute the missing values.....")
    print(f"the size of data before imputation: {data.shape}")
    count_missing = data.isna().sum().sum()
    print(f"number of missing values in data: {count_missing}")
    if count_missing > 0:
      if (self.save_dir / 'data_missing_value_filled.csv').exists() == False:
        data = fill_missinng_data(data, self.save_dir, aggregation=self.scale, seed=self.seed) # the index of data is Date_Time, while the output index is Sensor_Name
        save_data(data, save_dir=self.save_dir, file_name='data_missing_value_filled.csv', index=True, rewrite=self.rewrite)
      else:
        print(f"load {self.save_dir / 'data_missing_value_filled.csv'}.....")
        data = pd.read_csv(self.save_dir / 'data_missing_value_filled.csv')
        data.set_index(self.target_column, inplace=True)
      
      data.columns.name = self.time_column
      
      print(f"The size of data after imputation is {data.shape}")
    
    self.data = data
    return data

  def normalise_data(self, data=None):
    if data is None:
      data = self.data.copy()
      columns = self.data.columns
      index = self.data.index
    else:
      columns = data.columns
      index = data.index

    print("-"*50)
    print('normalising the data.....')
    if (self.save_dir / 'scaled_data.csv').exists() == False:
      if self.normalise is not None:
        print("scaling the data.....")
        if self.normalise == 'meanvariance':
          data = TimeSeriesScalerMeanVariance().fit_transform(data.values)
        elif self.normalise == 'minmax':
          data = TimeSeriesScalerMinMax().fit_transform(data.values)

        data = pd.DataFrame(data.squeeze(axis=-1), columns=columns, index=index)
        
        save_data(data, save_dir=self.save_dir, file_name='scaled_data.csv', index=True, index_label=self.target_column, rewrite=self.rewrite)
    else:
      print("loading the scaled data.....")
      data = pd.read_csv(self.save_dir / 'scaled_data.csv')
      data.set_index(self.target_column, inplace=True)
    
    print(f"the size of scaled data is {data.shape}")
    is_sorted = list(data.columns) == sorted(data.columns)
    print(f"if the scaled data sorted by time?: {is_sorted}")
    print(f"missing value in scaled data: {data.isna().sum().sum()}")
    
    self.data = data
    return data
    
  def optimal_k(self, data=None):
    if data is None:
      data = self.data.copy()
    
    print("-"*50)
    print('finding the optimal k.....')
    if (self.save_dir / 'best_k.xlsx').exists() == False:
      self.best_k, self.df_scores = find_optimal_k(
        data, self.save_dir, algorithm=self.algorithm, metric=self.model_configs['metric'],
        seed=self.seed, verbose=self.verbose)
      plot_best_k(self.df_scores, self.save_dir)
    else:
      self.df_scores = pd.read_excel(self.save_dir / 'best_k.xlsx')
      agg_best_k = self.df_scores.iloc[0]['Number_of_Clusters']
      df_scores = self.df_scores.sort_values(by='Number_of_Clusters')
      elbow_point = df_scores['Number_of_Clusters'].iloc[find_elbow(df_scores['Distortion'].values.tolist())]
      self.best_k = sorted(list(set([int(agg_best_k), int(elbow_point)])))
      print(f'the best k are {self.best_k}')
  
  def plot_results(self, data=None, filename='cluster_assignments.png', plot_data_center=False):
    if data is None:
      data = self.data.copy()

    print("-"*50)
    n_rows = self.best_k
    n_cols = 1
    sz = data.shape[1]

    plt.figure(figsize=(20, 5 * self.best_k))

    for yi in range(self.best_k):
      plt.subplot(n_rows, n_cols, yi + 1)
      for xx in data[self.y_pred == yi].values:
        plt.plot(xx.ravel(), "k-", alpha=.2)
      if self.algorithm in ['kshape', 'kmeans'] and plot_data_center == False:
        plt.plot(self.model.cluster_centers_[yi].ravel(), "r-")
      elif self.algorithm == 'birch' and plot_data_center == False:
        plt.plot(self.model.subcluster_centers_[yi].ravel(), "r-")
      plt.xlim(0, sz)
      plt.text(0.55, 0.9, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
      if yi == 0:
        plt.title("Cluster Assignments")
      
      if plot_data_center == True:
        mean_data = np.mean(data[self.y_pred == yi].values, axis=0)
        plt.plot(mean_data.ravel(), "orange")
    
    try:
      plt.savefig(self.save_dir / filename, dpi=800, bbox_inches='tight', pad_inches=0.1)
    except ValueError:
      plt.savefig(self.save_dir / filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

  def plot_data(self, data=None, fig_name=None):
    print(f'plot the {fig_name} .....')
    if self.time_span == 'normal':
      start_date, end_date = '~', '2019'
    elif isinstance(self.time_span, list) and len(self.time_span) == 2:
      start_date, end_date = self.time_span[0], self.time_span[1]
    elif isinstance(self.time_span, int):
      start_date, end_date = self.time_span, self.time_span
    
    if data is None:
      data = self.data
          
    plot_time_series_data_sensor(data, start_date, end_date, 
                                  save_path=self.save_dir, fig_name=fig_name,
                                  agg=['raw'], with_shadow_missing=[True])
  
  def load_image(self, save_dir=None, file_name=None, dpi=300):
    if save_dir is None:
      save_dir = self.save_dir
    img = mpimg.imread(save_dir / file_name)
    plt.figure(figsize=(img.shape[1]/dpi, img.shape[0]/dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

  def assign_clusters_and_save(self):
    print("-"*50)
    clusters = pd.DataFrame({
      self.target_column: self.data.index,
      'Clusters': self.model.labels_
    })

    # Merge with the original DataFrame
    df = self.data.reset_index().rename(columns={'index': self.target_column})
    df_clusters = pd.merge(df, clusters, on=self.target_column, how='left')

    df_raw = self.data_raw.reset_index().rename(columns={'index': self.target_column})
    df_clusters_raw = pd.merge(df_raw, clusters, on=self.target_column, how='left')

    df_raw_norm = self.data_raw_norm.reset_index().rename(columns={'index': self.target_column})
    df_clusters_raw_norm = pd.merge(df_raw_norm, clusters, on=self.target_column, how='left')

    sensor_locations_ = self.sensor_locations.copy()[[self.target_column, 'Latitude', 'Longitude']]
    sensor_locations_.drop_duplicates(inplace=True)

    data = pd.merge(df_clusters, sensor_locations_[[self.target_column, 'Latitude', 'Longitude']], how='left', on=self.target_column)
    data_raw = pd.merge(df_clusters_raw, sensor_locations_[[self.target_column, 'Latitude', 'Longitude']], how='left', on=self.target_column)
    data_raw_norm = pd.merge(df_clusters_raw_norm, sensor_locations_[[self.target_column, 'Latitude', 'Longitude']], how='left', on=self.target_column)
    
    self.df_clusters = data.dropna(subset=['Latitude', 'Longitude'])
    self.df_clusters_raw = data_raw.dropna(subset=['Latitude', 'Longitude'])
    self.df_clusters_raw_norm = data_raw_norm.dropna(subset=['Latitude', 'Longitude'])

    save_data(data, save_dir=self.save_dir, file_name='clusters.csv')
    save_data(data_raw, save_dir=self.save_dir, file_name='clusters_raw.csv')
    save_data(data_raw_norm, save_dir=self.save_dir, file_name='clusters_raw_norm.csv')

  def evaluation(self):
    print("-"*50)
    eva_scores = {
      'Silhouette Score': [silhouette_score_skl(self.data, self.model.labels_)],
      'Davies-Bouldin Score': [davies_bouldin_score(self.data, self.model.labels_)],
      'Calinski-Harabasz Score': [calinski_harabasz_score(self.data, self.model.labels_)]
    }

    self.df_results = pd.DataFrame(eva_scores)
    print_table(self.df_results)
    save_data(self.df_results, save_dir=self.save_dir, file_name='evaluation_scores.xlsx')

  def create_model(self, data=None):
    if data is None:
      data = self.data.copy()
    
    print("-"*50)
    print(f"{self.model_configs['metric']} {self.algorithm}")
    save_model = self.save_dir / 'model.pickle'

    if self.algorithm == 'kmeans':
      self.model = TimeSeriesKMeans(n_clusters=self.best_k, **self.model_configs)
    elif self.algorithm == 'kshape':
      self.model = KShape(n_clusters=self.best_k, **self.model_configs)
    elif self.algorithm == 'kernelkmeans':
      self.model = KernelKMeans(n_clusters=self.best_k, **self.model_configs)
    elif self.algorithm == 'autocluster': # TBD
      cluster = AutoCluster(logger=None) 
      result_dict = cluster.fit(data=data.values, **self.model_configs)
      predictions = cluster.predict(data.values, save_plot=False)
      print(result_dict["optimal_cfg"])
      print(Counter(predictions))
      print(cluster.get_trajectory())
      
    if not save_model.exists():
      print(f"Now is creating {save_model}")
      self.y_pred = self.model.fit_predict(data.values) # the data used for fitting and predicting 
                                                        # are the same, but can be further improved
      self.model.to_pickle(save_model)
    else:
      print(f"Now is loading {save_model}")
      self.model = self.model.from_pickle(save_model)
      self.y_pred = self.model.predict(data.values)

  def extract_features(self, drop_nan=True): # TBD
    print("-"*50)
    print("extract features from data.....")
    self.data.reset_index(inplace=True)
    df_melted = self.data.melt(id_vars=self.target_column, var_name=self.time_column, value_name=self.value_column)
    df_melted[self.time_column] = pd.to_datetime(df_melted[self.time_column], errors='coerce') 
    df_melted = df_melted.dropna(subset=[self.time_column])
    df_melted = df_melted.sort_values(by=[self.target_column, self.time_column])

    self.data = extract_features(df_melted, column_id=self.target_column, column_sort=self.time_column)

    # cfg = tsfel.get_features_by_domain()
    # features_list = []
    # for sensor in df_melted[self.target_column].unique():
    #   sensor_data = df_melted[df_melted[self.target_column] == sensor]
    #   features = tsfel.time_series_features_extractor(cfg, sensor_data[self.value_column])
    #   features[self.target_column] = sensor  # add sensor name to the features DataFrame
    #   features_list.append(features)

    # self.data = pd.concat(features_list, ignore_index=True)

    self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
    self.data = self.data.fillna(0)
    if drop_nan == True:
      print(f"the feature extracted from data has shape {self.data.shape} before drop NaNs")
      cols_to_drop = self.data.columns[(self.data == 0).all()]
      self.data = self.data.drop(columns=cols_to_drop)

    print(f"now the feature extracted from data has shape {self.data.shape}")
    save_data(self.data, save_dir=self.save_dir, file_name='extracted_features.xlsx', rewrite=self.rewrite)

  def dimensionality_reduction(self, data=None):
    if data is None:
      data = self.data.copy()

    print('-'*50)
    print(f'Now applying {self.dim_reduction} to data')
    if self.dim_reduction == 'PCA':
      pca = PCA(n_components=0.95)
      # pca = PCA(n_components=15)
    elif self.dim_reduction == 'IPCA':
      pca = IncrementalPCA(n_components=0.95)
    else:
      pass # more methods

    transformed_data = pca.fit_transform(data)
    data = pd.DataFrame(transformed_data, index=data.index)
    plot_pca_heatmap(pca, data.index, save_dir=self.save_dir)
    print(f'the data size after dimensionality reduction: {data.shape}')
    save_data(data, save_dir=self.save_dir, file_name='reduced_data.csv', index=True)

    self.data = data
    
    return data
    
  def process_data(self): # On going
    # process the data before feeding them into model

    # the data should be in (M * N) shape, where M is the number of samples, 
    # N is number of features (timestamps)
    self.data = self.data.pivot(index=self.target_column, columns=self.time_column, values=self.value_column)

    self.data.columns = pd.to_datetime(self.data.columns)
    # TBD: the error may occur if agg first, need further debugging
    
    data_cut = self.select_time_span(data=self.data)
    self.plot_data(data=data_cut, fig_name='plot_raw_data') # plot raw data

    for operation in self.order_of_impute_agg_norm.split("_"):
      if operation == "impute":
        self.impute_data() # impute data first then select time span or vice versa?
        self.select_time_span(keep_index=data_cut.index) # select specific time span 
        self.plot_data(fig_name='plot_imputed_data')
      elif operation == "agg":
        self.aggregation()
        self.plot_data(fig_name='plot_aggregated_data')
      elif operation == "norm":
        self.data_raw = self.data

        self.normalise_data()
        self.plot_data(fig_name='plot_normalised_data')

        self.data_raw_norm = self.data
    
    if self.feature_extraction is not None:
        self.extract_features()

    if self.dim_reduction is not None:
      self.dimensionality_reduction()

  def training_each_k(self):
    best_k = self.best_k
    for self.best_k in best_k: # create model for each k
      self.best_k = int(self.best_k)

      # update save_dir
      self.save_dir = set_path(self.save_dir / f"best_k_{self.best_k}")

      self.create_model() # create model
      self.plot_results() # plot the results
      self.plot_results(self.data_raw_norm, 'cluster_assignments_raw_norm.png', plot_data_center=True) # plot the results
      self.evaluation()
      self.assign_clusters_and_save() # assign the labels (clusters) back to the original DataFrame
      plot_map(self.df_clusters, self.save_dir) # plot the clusters on the map
  
      # reset save_dir
      self.set_save_dir(self.root_save_dir)
    
    self.best_k = best_k

  def offline_training(self):
    self.set_save_dir(self.root_save_dir)
    self.save_configs()
    if (self.save_dir / 'df_final.csv').exists() == False:
      self.process_data() # prepare data
      save_data(self.data, save_dir=self.save_dir, file_name='df_final.csv', header=True, index=True)
    else:
      print("loading the preprocessed data.....")
      self.data = pd.read_csv(self.save_dir / 'df_final.csv')
      self.data.set_index(self.target_column, inplace=True)
    
    self.optimal_k() # find optimal k (for k-means)
    
    self.training_each_k()
    
  def online_training(self, files): # TBD
    self.set_save_dir(self.root_save_dir)
    self.save_configs()

    save_model = self.save_dir / 'model.pickle'
    
    self.model = Birch(**self.model_configs)
    
    original_save_dir = self.save_dir
    for file_name in files: 
      print(file_name)

      # update save_dir, so that the results for each year are saved in their respective year-named folders
      start_index = file_name.index('_') + 1
      end_index = file_name.index('.')
      time_span = file_name[start_index:end_index]
      save_dir = original_save_dir / f"{time_span}"
      self.save_dir = set_path(save_dir)

      if (self.save_dir / 'df_final.csv').exists() == False:
        self.process_data() # prepare data
        save_data(self.data, save_dir=self.save_dir, file_name='df_final.csv', header=True, index=True)
      else:
        print("loading the preprocessed data.....")
        self.data = pd.read_csv(self.save_dir / 'df_final.csv')
        self.data.set_index(self.target_column, inplace=True)
      
      if not save_model.exists():
        print(f"Now is creating {save_model}")
        self.model.partial_fit(self.data)
        self.y_pred = self.model.predict(self.data)
      else:
        print(f"Now is loading {save_model}")
        with open(save_model, 'rb') as f:
          self.model = pickle.load(f)
        self.model.partial_fit(self.data)
        self.y_pred = self.model.predict(self.data)
      
      with open(save_model, 'wb') as f:
        pickle.dump(self.model, f)
      
      save_submodel = self.save_dir / f"model_{time_span}.pickle"
      
      with open(save_submodel, 'wb') as f:
        pickle.dump(self.model, f)

      self.best_k = len(np.unique(self.model.labels_))
      print(f"the data split into {self.best_k} clusters.")

      self.plot_results()
      self.evaluation()
      self.assign_clusters_and_save() # assign the labels (clusters) back to the original DataFrame
      plot_map(self.df_clusters, self.save_dir) # plot the clusters on the map
   
    self.set_save_dir(self.root_save_dir)
      