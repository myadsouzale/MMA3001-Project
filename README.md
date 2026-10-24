# MMA3001-Project
# Building Occupancy Pattern Analysis and Prediction

**MMA3001 – Numerical Methods and Machine Learning**  
**Monash University | 2026**

## 1. Project Overview

This project investigates the research question:

**To what extent can historical occupancy data be used to identify and predict recurring weekly occupancy patterns across different building spaces?**

The project uses historical building occupancy sensor data to identify patterns in how spaces are occupied throughout the day and week. It also evaluates whether these patterns can be used to predict future hourly occupancy levels.

Understanding building occupancy is relevant to engineering applications such as space utilisation, building operations, heating, ventilation and air-conditioning (HVAC) scheduling, and energy management. Reliable occupancy estimates could help inform more efficient use of building facilities.

The computational workflow is implemented in Python and includes data cleaning, time-weighted hourly aggregation, exploratory data analysis, predictive modelling, validation, sensitivity analysis and computational performance evaluation.

## 2. Project Objectives

The main objectives are to:

1. Process and clean a large historical occupancy dataset.
2. Convert irregularly sampled sensor observations into representative hourly occupancy values.
3. Identify recurring daily and weekly occupancy patterns across different spaces.
4. Develop and compare predictive models for hourly occupancy.
5. Evaluate prediction accuracy using chronologically separated training, validation and testing data.
6. Investigate the effects of temporal variability and hourly sensor coverage on model performance.
7. Compare computational accuracy, runtime, memory usage and model complexity.

## 3. Dataset

The original dataset contains **9,091,732 occupancy sensor observations** covering five building spaces.

The available timestamps extend from 24 November 2023 to 22 April 2026.

The primary raw dataset is:

`5occupancySensor_MayToDec2024_9MRows.csv`

The raw data contains occupancy-related information including:

| Variable | Description |
|---|---|
| `deviceid` | Identifier associated with the sensor data source |
| `floorspaceid` | Unique identifier for a monitored building space |
| Timestamp | Time at which the observation was recorded |
| `occupancystatus` | Reported occupancy state |
| `headcount` | Estimated number of people occupying the space |

The raw dataset contains irregularly spaced observations, meaning that directly averaging recorded headcounts could overrepresent periods with frequent sensor updates.

To address this, time-weighted hourly occupancy values are calculated.

### Data Cleaning

The cleaning process:

- Reads the large dataset in chunks to reduce memory requirements.
- Processes timestamps and occupancy information.
- Removes invalid records where applicable.
- Removes duplicate observations.
- Produces a cleaned dataset for subsequent analysis.

Of the 9,091,732 original records, four duplicate records were removed, leaving **9,091,728 cleaned observations**.

### Hourly Aggregation

Occupancy observations are converted into hourly statistics, including:

- Mean observed headcount.
- Time-weighted mean headcount.
- Maximum observed headcount.
- Number of observations.
- Temporal coverage of each hour.
- Whether the hour satisfies the selected coverage threshold.

Time-weighted occupancy accounts for the duration represented by each sensor observation, with forward propagation limited to 10 minutes.

The resulting dataset contains **62,325 hourly observations** across the five monitored spaces.

The original analysis considered hours with at least 80% temporal coverage to be reliable. A subsequent sensitivity analysis investigated thresholds of 50%, 60%, 80% and 90%.

## 4. Computational Methodology

### 4.1 Exploratory Data Analysis

Historical occupancy patterns are investigated using:

- Weekly occupancy heatmaps.
- Average hourly occupancy profiles.
- Weekday comparisons.
- Monthly occupancy trends.
- Comparisons between monitored spaces.

These visualisations are used to identify recurring occupancy patterns and differences in space utilisation.

### 4.2 Predictive Models

Three prediction methods are compared.

**Space Mean Baseline**

Predicts occupancy using the historical average headcount for each space.

This provides a simple reference model that does not explicitly account for daily or weekly variation.

**Historical Weekly-Pattern Model**

Predicts occupancy using historical average headcounts grouped by building space, day of the week and hour of the day.

Where a particular combination is unavailable, the model uses progressively broader historical averages.

This method is interpretable and directly represents recurring weekly occupancy behaviour.

**Random Forest Regression**

A machine-learning model that predicts occupancy using calendar-related features and building-space identifiers.

Features include:

- Building space.
- Day of the week.
- Hour of the day.
- Month.

The original Random Forest configuration uses 150 trees, a maximum tree depth of 12 and a minimum of 10 samples per leaf.

### 4.3 Model Validation

The dataset is divided chronologically into training, validation and testing periods.

Chronological splitting is used rather than random splitting to reduce the risk of future observations influencing the training process.

Model performance is evaluated using:

**Mean Absolute Error (MAE)**

The average absolute difference between predicted and observed hourly headcounts.

**Root Mean Squared Error (RMSE)**

An error metric that gives greater weight to larger prediction errors.

Both metrics are expressed in people. Lower values indicate better predictive performance.

Validation data is used to compare modelling approaches and make design decisions. The held-out testing period provides a subsequent evaluation of predictive performance.

### 4.4 Temporal Stability Analysis

Prediction errors are examined across different months and building spaces to investigate whether model performance remains consistent over time.

Four building spaces are used for the main temporal stability and optimisation comparisons. The fifth space is excluded from these comparisons because its observations do not extend into the later validation and testing periods.

### 4.5 Sensitivity Analysis

A sensitivity analysis investigates minimum hourly coverage thresholds of:

- 50%
- 60%
- 80%
- 90%

All configurations are evaluated using the same high-coverage validation and testing observations to ensure a fair comparison.

The analysis examines how the amount of available historical training data affects predictive accuracy.

### 4.6 Computational Performance and Optimisation

Computational performance is evaluated through:

- Comparison of Random Forest configurations containing 30, 75 and 150 trees.
- Five repeated training and prediction runtime measurements.
- Preprocessing runtime measurements.
- Process-level memory profiling.
- Comparison of accuracy and computational complexity.

These experiments support the selection of an appropriate computational method for the engineering application.

## 5. Main Findings

The historical occupancy data demonstrates recurring patterns associated with the day of the week and time of day, although these patterns differ between monitored spaces.

The historical weekly-pattern model achieved better validation accuracy than the Random Forest configurations tested, while requiring substantially less computational time.

In the coverage sensitivity experiment, reducing the training coverage threshold from 80% to 50% increased available training observations from 23,140 to 29,335.

For the weekly-pattern model, validation MAE decreased from 0.7829 to 0.7399 people, representing an improvement of approximately 5.5%.

The Random Forest achieved lower errors in some held-out testing comparisons, but its validation performance was less consistent.

The computational performance analysis identified data cleaning as the longest preprocessing operation and hourly aggregation as the most memory-intensive operation.

Overall, the results suggest that relatively simple historical averaging methods can provide useful and interpretable predictions of recurring occupancy patterns, without necessarily requiring more complex machine-learning models.

## 6. Repository Structure

```text
MMA3001 Building Occupancy Project/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data files/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── clean_occupancy_data.py
│   ├── check_occupancy_dates.py
│   ├── check_headcount.py
│   ├── aggregate_occupancy_hourly.py
│   ├── analyse_weekly_patterns.py
│   ├── predict_occupancy.py
│   ├── evaluate_temporal_stability.py
│   ├── optimise_model_performance.py
│   ├── benchmark_models.py
│   ├── profile_pipeline.py
│   └── sensitivity_analysis.py
│
├── results/
│   ├── optimisation/
│   ├── temporal_stability/
│   └── sensitivity/
│
├── docs/
│   ├── data_dictionary.md
│   └── html/
│
└── tests/
```

The `src` directory contains the Python analysis scripts. The `results` directory contains generated figures, numerical results and performance comparisons. The `docs` directory contains supporting documentation.

The `tests` directory is intended for automated verification of the computational implementation.

## 7. Installation and Setup

### Requirements

- Python 3
- pandas
- NumPy
- Matplotlib
- scikit-learn
- psutil

Development and documentation tools include pytest and pdoc.

### Creating a Virtual Environment

Open a terminal in the project directory and run:

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Input Data

Place the original occupancy dataset in:

`data files/raw/`

The raw and processed occupancy datasets are excluded from Git version control because of their size and potential data-sharing restrictions.

Access to the original dataset is required to reproduce the complete preprocessing workflow. Users without access to the dataset can inspect the source code and any committed summary results but cannot reproduce the complete analysis from raw observations.

## 8. Running the Project

Run commands from the project root directory.

### Step 1 — Clean the Data

```powershell
python -u src/clean_occupancy_data.py
```

### Step 2 — Generate Hourly Occupancy Statistics

```powershell
python -u src/aggregate_occupancy_hourly.py
```

### Step 3 — Analyse Weekly Occupancy Patterns

```powershell
python -u src/analyse_weekly_patterns.py
```

### Step 4 — Train and Evaluate Predictive Models

```powershell
python -u src/predict_occupancy.py
```

### Step 5 — Evaluate Temporal Stability

```powershell
python -u src/evaluate_temporal_stability.py
```

### Step 6 — Compare Model Configurations

```powershell
python -u src/optimise_model_performance.py
```

### Step 7 — Benchmark Model Runtime

```powershell
python -u src/benchmark_models.py
```

### Step 8 — Profile Preprocessing Performance

```powershell
python -u src/profile_pipeline.py
```

This script reruns data cleaning and hourly aggregation, potentially overwriting their existing output files.

### Step 9 — Perform Coverage Sensitivity Analysis

```powershell
python -u src/sensitivity_analysis.py
```

Generated results are saved in the `results` directory and its subdirectories.

## 9. Documentation and Testing

The project includes supporting data documentation describing the meanings, formats and units of the principal variables.

HTML source-code documentation can be generated using pdoc:

```powershell
python -m pdoc -o docs/html src
```

Automated tests are intended to verify key numerical calculations, data-processing assumptions, prediction fallbacks and input handling.

Once implemented, tests can be executed using:

```powershell
python -m pytest tests/ -v
```

Test outcomes should be recorded in a test report.

## 10. Assumptions and Limitations

The computational workflow relies on several assumptions:

- Reported sensor headcounts are treated as estimates of actual occupancy.
- Time-weighted hourly averages are used to represent occupancy over an hour.
- Sensor observations are not propagated indefinitely through periods without measurements.
- Historical calendar patterns are assumed to contain information relevant to future occupancy.
- Building-space identifiers are used without requiring a complete physical room mapping.
- Missing observations are not automatically interpreted as zero occupancy.

Important limitations include incomplete temporal coverage, possible sensor measurement errors, differences between monitored spaces and changes in occupancy behaviour over time.

The models primarily use historical occupancy and calendar information. They do not explicitly incorporate holidays, timetables, building events or other external drivers.

Prediction accuracy on historical held-out data does not guarantee equivalent performance under substantially different future operating conditions.

## 11. Engineering Applications

Potential applications of the computational approach include:

- Identifying recurring periods of high and low building occupancy.
- Supporting space utilisation assessments.
- Informing building operations and facility scheduling.
- Providing occupancy estimates that could contribute to future HVAC and energy-management studies.

The model is intended as a decision-support method rather than a validated real-time building control system.

## 12. AI Use

ChatGPT was used to support programming, computational methodology development, debugging, interpretation of results and preparation of documentation.

AI-generated suggestions and code require verification against the dataset, program outputs, engineering assumptions and assessment requirements.

A detailed reflection on the extent of AI contribution, verification methods, limitations and independent engineering decisions is provided in the accompanying project report.

## 13. Licensing and Data Availability

The licensing conditions for project source code are specified in the repository's `LICENSE` file.

The original occupancy dataset is not redistributed through this repository. Access and reuse remain subject to the conditions imposed by the dataset provider.

## 14. Project Status

The data analysis, model comparisons, performance benchmarking and sensitivity analysis have been completed.

Repository documentation, automated software testing and final assessment materials are being finalised.