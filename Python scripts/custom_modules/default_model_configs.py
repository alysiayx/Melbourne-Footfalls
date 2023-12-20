# from autocluster import get_evaluator, MetafeatureMapper

def get_default_model_configs(algorithm, seed):
  if algorithm == 'kmeans':
    default_model_configs = {
      'metric': 'euclidean',
      'n_init': 10, 
      'max_iter': 300,
      'init': 'k-means++',
      "random_state": seed
    }
  elif algorithm == 'kernelkmeans':
    default_model_configs = {
      'kernel': 'gak',
      'kernel_params': {"sigma": "auto"},
      'max_iter': 300,
      "random_state": seed
    }
  elif algorithm == 'birch':
    default_model_configs = {
      "branching_factor": 50,
      "threshold": 0.5,
      "n_clusters": 9,
    }
  elif algorithm == 'autocluster':
    default_model_configs = {
      "cluster_alg_ls": [
          'KMeans', 'GaussianMixture', 'MiniBatchKMeans', 'KernelKMeans', 'KShape',
          'DBSCAN', 'BIRCH', 'Birch', 'AgglomerativeClustering', 'SpectralClustering',
          'MeanShift', 'AffinityPropagation'
      ], 
      "dim_reduction_alg_ls": [
          'NullModel', 'TSNE', 'PCA', 'IncrementalPCA', 'KernelPCA', 'FastICA', 'TruncatedSVD', 'LDA'
      ],
      "optimizer": 'smac',
      "n_evaluations": 40,
      "run_obj": 'quality',
      "seed": 27,
      "cutoff_time": 10,
      "preprocess_dict": {
          "numeric_cols": list(range(2)),
          "categorical_cols": [],
          "ordinal_cols": [],
          "y_col": []
      },
      "evaluator": get_evaluator(evaluator_ls = ['silhouetteScore', 
                                                'daviesBouldinScore', 
                                                'calinskiHarabaszScore'], 
                                weights = [1, 1, 1], 
                                clustering_num = None, 
                                min_proportion = .01, 
                                min_relative_proportion='default'),
      "n_folds": 3,
      # "warmstart": False,
      "warmstart": True,
      "warmstart_datasets_dir": 'experiments/metaknowledge/benchmark_silhouette',
      "warmstart_metafeatures_table_path": 'experiments/metaknowledge/benchmark_silhouette_metafeatures_table.csv',
      "warmstart_n_neighbors": 10,
      "warmstart_top_n": 3,
      "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
      "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
      "categorical_metafeatures": [],
      "verbose_level": 1,
    }
  return default_model_configs