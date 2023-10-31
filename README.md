# Melbourne Footfall Analysis and Modelling

## Basic Data Analysis

### Data Format Analysis
📊 **Notebook**: [`data_format_analysis.ipynb`](./Python%20scripts/basic_analysis/data_format_analysis.ipynb)
   - Get acquainted with the data format we choose for further study

### Imputation Analysis
📈 **Notebook**: [`imputation_analysis.ipynb`](./Python%20scripts/basic_analysis/imputation_analysis.ipynb)
   - Experienemnts of imputation

## Data Pre-Processing

### How to Run:
🔄 **Notebook**: Simply download the data we have uploaded: [Data (20230918)](./Data%20(20230918)/), and then execute the [Melbourne Footfall Data Preprocessing Notebook](./Python%20scripts/Melbourne_footfall_data_preprocessing.ipynb) to perform the necessary data pre-processing.

The latest data also can be downloaded from:
- [Pedestrian Counting System (counts per hour)](https://melbournetestbed.opendatasoft.com/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/information/)

- [Pedestrian Counting System - Sensor Locations](https://melbournetestbed.opendatasoft.com/explore/dataset/pedestrian-counting-system-sensor-locations/information/)

### Pre-Processing Steps:
1. **Duplicate Data Handling**: 
   - Purpose: Maintain data integrity and accuracy.
   - Action: Identify and eliminate records with duplicate sensor IDs, location IDs, or geo-locations.

2. **Sensor ID Unification**: 
   - Purpose: Achieve consistency across datasets, especially for data collected in 2023.
   - Action: Augment records from 2023 that only contain Location IDs with the corresponding sensor names and geo-locations.

the preprocessed data are saved in [data_preprocessed](./data_preprocessed/)
   - [1. merged_peds_data_hist_curr](./data_preprocessed/1.%20merged_peds_data_hist_curr/) stores the data for offline learning, please unzip [footfall_merged.csv.zip](./data_preprocessed/1.%20merged_peds_data_hist_curr/footfall_merged.csv.zip) before executing the notebooks.
   - [1. merged_peds_data_hist_curr](./data_preprocessed/1.%20merged_peds_data_hist_curr/).
   - [4. final_group](./data_preprocessed/4.%20final_group/) stores the data for online learning.

## Footfall Modelling

Explore pedestrian patterns in Melbourne.

### Guided Modelling:
📘 **For Beginners and Detailed Instructions**:
   - Please refer to the step-by-step guide available in the [`guide_how_to_run_the_model.ipynb`](./Python%20scripts/guide_how_to_run_the_model.ipynb) notebook.

### Direct Modelling:
🚀 **For Experienced Users**:
   - If you are already familiar with the process, you can jump straight into the main modelling notebook: [`Melbourne_footfall_modelling.ipynb`](./Python%20scripts/Melbourne_footfall_modelling.ipynb).
