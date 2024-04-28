# Melbourne Footfall Analysis and Modelling

## Dependencies

**Setting Up the Environment:**

Open your terminal or command prompt.
Use the `cd` command to change directory to the location where you downloaded the "Melbourne-Footfalls" project. For example:

```bash
    cd /path/to/Melbourne-Footfalls
```
Replace `/path/to/Melbourne-Footfalls` with the actual path to the directory where you downloaded the project.

- **Option 1:** Using `requirements.txt`
  - Create a virtual environment
    - using venv or virtualenv:
    ```bash
    python -m venv <env_name>
    source <env_name>/bin/activate   # On Unix/Linux
    <env_name>\Scripts\activate.bat  # On Windows
    ```
    
    - using conda:
      ```bash
      conda create --name <env_name> python=3.9 --file requirements.txt
      conda activate <env_name>
      ```
    
    replace `<env_name>` with the name of virtual environement you specified.

  - Execute the following command in the terminal to install the project dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- **Option 2:** Using `pyproject.toml`
  - Execute the following command in the terminal to install the project dependencies:
    ```bash
    pip install poetry
    poetry install
    poetry shell
    ```

## Basic Data Analysis

### Data Format Analysis

📊 **Notebook:** Access the analysis via [`data_format_analysis.ipynb`](./Python%20scripts/basic_analysis/data_format_analysis.ipynb).
- Understand the chosen data format for our analysis.

### Imputation Analysis

📈 **Notebook:** Explore through [`imputation_analysis.ipynb`](./Python%20scripts/basic_analysis/imputation_analysis.ipynb).
- Conduct experiments related to data imputation.

## Data Pre-Processing

### How to Execute:

🔄 **Notebook:** Download the dataset [Data (20230918)](./Data%20(20230918)/) and process it using [`Melbourne_footfall_data_preprocessing.ipynb`](./Python%20scripts/Melbourne_footfall_data_preprocessing.ipynb).

**Additional Data Sources:**
- [Pedestrian Counting System (counts per hour)](https://melbournetestbed.opendatasoft.com/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/information/)
- [Pedestrian Counting System - Sensor Locations](https://melbournetestbed.opendatasoft.com/explore/dataset/pedestrian-counting-system-sensor-locations/information/)

### Pre-Processing Steps:

1. **Duplicate Data Handling**:
   - **Objective:** Ensure data integrity and accuracy.
   - **Action:** Remove records with duplicated sensor IDs, location IDs, or geo-locations.
2. **Sensor ID Unification**:
   - **Objective:** Standardize datasets, focusing on 2023 data.
   - **Action:** Complement 2023 records, which have only Location IDs, with corresponding sensor names and geo-locations.

**Preprocessed Data Storage:**
- [1. merged_peds_data_hist_curr](./data_preprocessed/1.%20merged_peds_data_hist_curr/): Contains data for offline learning. Unzip [footfall_merged.csv.zip](./data_preprocessed/1.%20merged_peds_data_hist_curr/footfall_merged.csv.zip) before using the notebooks.
- [4. final_group](./data_preprocessed/4.%20final_group/): Contains data for online learning. Data is segmented to ensure completeness and limit missing data to no more than 50%.

## Footfall Modelling

**Objective:** Analyze pedestrian traffic patterns in Melbourne.

### Guided Modelling:

📘 **For Beginners or Detailed Guidance**:
- Follow the step-by-step instructions in [`guide_how_to_run_the_model.ipynb`](./Python%20scripts/guide_how_to_run_the_model.ipynb).

### Direct Modelling:

🚀 **For Advanced Users**:
- Proceed directly to the main modelling notebook: [`Melbourne_footfall_modelling.ipynb`](./Python%20scripts/Melbourne_footfall_modelling.ipynb).

### Future Developments:

- Integration of AutoML is pending.
- Plans to combine data preparation with model training and execution, rather than pre-processing for online learning separately.