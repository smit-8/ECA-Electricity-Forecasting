# Electricity Consumption Analysis / Forecasting (ECA)

A Python project that analyses household electricity consumption, trains machine learning models to forecast hourly usage, and presents the results in a Streamlit dashboard.

## Project structure
```
ECA-Electricity-Forecasting/
|-- notebooks/    10 Jupyter notebooks (data loading to dashboard preparation)
|-- data/
|   |-- raw/        original dataset (download separately, see data/raw/README.md)
|   |-- processed/  cleaned, hourly, train/test and prediction files
|   |-- processed/dashboard/  files used by the dashboard
|-- models/       best_electricity_forecasting_model.joblib
|-- dashboard/    11_streamlit_dashboard.py
|-- figures/      saved charts
|-- reports/      notes
|-- requirements.txt
```

## Notebooks (run in order)
1. 01_data_loading  2. 02_data_cleaning  3. 03_hourly_aggregation  4. 04_eda  5. 05_feature_engineering
6. 06_train_test_split  7. 07_model_training  8. 08_model_evaluation  9. 09_forecasting_demo  10. 10_dashboard_preparation

## How to run the dashboard
```
pip install -r requirements.txt
python -m streamlit run dashboard/11_streamlit_dashboard.py
```
The dashboard works with the processed files already included. The raw dataset is needed only to re-run notebooks 01 and 02.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Joblib.

