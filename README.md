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
🔄 **Notebook**: Simply execute the [`Melbourne_footfall_data_preprocessing.ipynb`](./Python%20scripts/Melbourne_footfall_data_preprocessing.ipynb) notebook to perform the necessary data pre-processing.

### Pre-Processing Steps:
1. **Duplicate Data Handling**: 
   - Purpose: Maintain data integrity and accuracy.
   - Action: Identify and eliminate records with duplicate sensor IDs, location IDs, or geo-locations.

2. **Sensor ID Unification**: 
   - Purpose: Achieve consistency across datasets, especially for data collected in 2023.
   - Action: Augment records from 2023 that only contain Location IDs with the corresponding sensor names and geo-locations.

## Footfall Modelling

Explore pedestrian patterns in Melbourne.

### Guided Modelling:
📘 **For Beginners and Detailed Instructions**:
   - Please refer to the step-by-step guide available in the [`guide_how_to_run_the_model.ipynb`](./Python%20scripts/guide_how_to_run_the_model.ipynb) notebook.

### Direct Modelling:
🚀 **For Experienced Users**:
   - If you are already familiar with the process, you can jump straight into the main modelling notebook: [`Melbourne_footfall_modelling.ipynb`](./Python%20scripts/Melbourne_footfall_modelling.ipynb).
