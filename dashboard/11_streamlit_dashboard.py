import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electricity Consumption Analysis",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DASHBOARD_DIR = BASE_DIR / "data" / "processed" / "dashboard"
MODEL_PATH = BASE_DIR / "models" / "best_electricity_forecasting_model.joblib"
HOURLY_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "hourly_electricity_consumption.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

@st.cache_data
def load_dashboard_data():
    kpis = json.loads(
        (DASHBOARD_DIR / "dashboard_kpis.json").read_text(
            encoding="utf-8"
        )
    )

    hourly_pattern = pd.read_csv(
        DASHBOARD_DIR / "hourly_pattern.csv"
    )

    day_pattern = pd.read_csv(
        DASHBOARD_DIR / "day_pattern.csv"
    )

    weekend_pattern = pd.read_csv(
        DASHBOARD_DIR / "weekend_pattern.csv"
    )

    month_pattern = pd.read_csv(
        DASHBOARD_DIR / "month_pattern.csv"
    )

    top_consumption = pd.read_csv(
        DASHBOARD_DIR / "top_consumption.csv",
        parse_dates=["DateTime"]
    )

    prediction_dashboard = pd.read_csv(
        DASHBOARD_DIR / "prediction_dashboard.csv",
        parse_dates=["DateTime"]
    )

    model_performance = pd.read_csv(
        DASHBOARD_DIR / "model_performance.csv"
    )

    hourly_df = pd.read_csv(
        HOURLY_DATA_PATH,
        parse_dates=["DateTime"]
    ).set_index("DateTime").sort_index()

    return (
        kpis,
        hourly_pattern,
        day_pattern,
        weekend_pattern,
        month_pattern,
        top_consumption,
        prediction_dashboard,
        model_performance,
        hourly_df
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

try:
    (
        kpis,
        hourly_pattern,
        day_pattern,
        weekend_pattern,
        month_pattern,
        top_consumption,
        prediction_dashboard,
        model_performance,
        hourly_df
    ) = load_dashboard_data()

    model = load_model()

except FileNotFoundError as error:
    st.error(
        "A required project file was not found."
    )
    st.code(str(error))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Consumption Analysis",
        "Forecasting",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Electricity Consumption Analysis\n\n"
    "Dataset: Household Electric Power Consumption\n\n"
    "Forecast target: Next-hour electricity consumption"
)


# ============================================================
# HEADER
# ============================================================

st.title("⚡ Electricity Consumption Analysis")
st.caption(
    "Data Analysis • Machine Learning • Next-Hour Forecasting"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "This dashboard analyzes household electricity consumption "
        "and presents a machine-learning model for next-hour "
        "electricity consumption forecasting."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Consumption",
        f'{kpis["Average Consumption"]:.3f}'
    )

    col2.metric(
        "Peak Consumption",
        f'{kpis["Peak Consumption"]:.3f}'
    )

    col3.metric(
        "Best Model",
        kpis["Best Model"]
    )

    col4.metric(
        "Model R²",
        f'{kpis["Best Model R2"]:.3f}'
    )

    st.markdown("---")

    st.subheader("Key Project Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f'{kpis["Best Model MAE"]:.4f}'
    )

    col2.metric(
        "RMSE",
        f'{kpis["Best Model RMSE"]:.4f}'
    )

    col3.metric(
        "MAE Improvement",
        f'{kpis["MAE Improvement Percent"]:.2f}%'
    )

    st.markdown("---")

    st.subheader("Overall Consumption Trend")

    trend_df = hourly_df[
        ["Electricity_Consumption"]
    ].copy()

    trend_df = trend_df.resample("D").mean().reset_index()

    fig = px.line(
        trend_df,
        x="DateTime",
        y="Electricity_Consumption",
        title="Daily Average Electricity Consumption"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Consumption",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Peak Consumption")

    st.write(
        f'Highest recorded hourly consumption: '
        f'**{kpis["Peak Consumption"]:.3f}**'
    )

    st.write(
        f'Peak timestamp: **{kpis["Peak Consumption Time"]}**'
    )


# ============================================================
# CONSUMPTION ANALYSIS
# ============================================================

elif page == "Consumption Analysis":

    st.header("📊 Consumption Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Hourly",
            "Day of Week",
            "Weekday vs Weekend",
            "Monthly"
        ]
    )

    with tab1:

        st.subheader("Average Electricity Consumption by Hour")

        fig = px.line(
            hourly_pattern,
            x="Hour",
            y="Average_Consumption",
            markers=True,
            title="Average Consumption by Hour"
        )

        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        peak_hour = hourly_pattern.loc[
            hourly_pattern["Average_Consumption"].idxmax()
        ]

        low_hour = hourly_pattern.loc[
            hourly_pattern["Average_Consumption"].idxmin()
        ]

        col1, col2 = st.columns(2)

        col1.metric(
            "Highest Average Hour",
            f'{int(peak_hour["Hour"]):02d}:00'
        )

        col2.metric(
            "Lowest Average Hour",
            f'{int(low_hour["Hour"]):02d}:00'
        )

    with tab2:

        st.subheader("Average Consumption by Day of Week")

        fig = px.bar(
            day_pattern,
            x="DayName",
            y="Average_Consumption",
            title="Average Consumption by Day"
        )

        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Average Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab3:

        st.subheader("Weekday vs Weekend")

        fig = px.bar(
            weekend_pattern,
            x="index",
            y="Average_Consumption",
            title="Weekday vs Weekend Consumption"
        )

        fig.update_layout(
            xaxis_title="Day Type",
            yaxis_title="Average Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab4:

        st.subheader("Average Consumption by Month")

        fig = px.line(
            month_pattern,
            x="Month",
            y="Average_Consumption",
            markers=True,
            title="Monthly Average Consumption"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Top 10 Highest Consumption Periods")

    display_top = top_consumption.head(10).copy()

    display_top["DateTime"] = (
        display_top["DateTime"]
        .dt.strftime("%Y-%m-%d %H:%M")
    )

    display_top = display_top.rename(
        columns={
            "DateTime": "Date & Time",
            "Electricity_Consumption": "Consumption"
        }
    )

    st.dataframe(
        display_top,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header("🔮 Next-Hour Electricity Forecast")

    st.write(
        "Select a historical timestamp. The trained model will "
        "use the available historical features to predict "
        "electricity consumption for the following hour."
    )

    forecast_df = hourly_df[
        ["Electricity_Consumption"]
    ].copy()

    forecast_df["Hour"] = forecast_df.index.hour
    forecast_df["DayOfWeek"] = forecast_df.index.dayofweek
    forecast_df["Month"] = forecast_df.index.month
    forecast_df["Quarter"] = forecast_df.index.quarter
    forecast_df["IsWeekend"] = (
        forecast_df["DayOfWeek"] >= 5
    ).astype(int)

    forecast_df["Lag_1h"] = (
        forecast_df["Electricity_Consumption"].shift(1)
    )

    forecast_df["Lag_24h"] = (
        forecast_df["Electricity_Consumption"].shift(24)
    )

    forecast_df["Lag_168h"] = (
        forecast_df["Electricity_Consumption"].shift(168)
    )

    forecast_df["Rolling_24h"] = (
        forecast_df["Electricity_Consumption"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    forecast_df["Rolling_168h"] = (
        forecast_df["Electricity_Consumption"]
        .shift(1)
        .rolling(168)
        .mean()
    )

    feature_columns = [
        "Hour",
        "DayOfWeek",
        "Month",
        "Quarter",
        "IsWeekend",
        "Lag_1h",
        "Lag_24h",
        "Lag_168h",
        "Rolling_24h",
        "Rolling_168h"
    ]

    valid_times = forecast_df.dropna(
        subset=feature_columns
    ).index

    selected_time = st.selectbox(
        "Select historical timestamp",
        valid_times[-1000:]
    )

    selected_features = forecast_df.loc[
        [selected_time],
        feature_columns
    ]

    prediction = float(
        model.predict(selected_features)[0]
    )

    target_time = (
        selected_time + pd.Timedelta(hours=1)
    )

    actual = None

    if target_time in forecast_df.index:
        actual = float(
            forecast_df.loc[
                target_time,
                "Electricity_Consumption"
            ]
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Time",
        selected_time.strftime("%Y-%m-%d %H:%M")
    )

    col2.metric(
        "Predicted Next Hour",
        f"{prediction:.3f}"
    )

    if actual is not None:

        col3.metric(
            "Actual Next Hour",
            f"{actual:.3f}"
        )

        st.metric(
            "Absolute Error",
            f"{abs(actual - prediction):.3f}"
        )

    st.markdown("---")

    st.subheader("Forecast vs Recent Consumption")

    recent_start = selected_time - pd.Timedelta(hours=24)
    recent_end = target_time

    recent_df = hourly_df.loc[
        recent_start:recent_end,
        ["Electricity_Consumption"]
    ].reset_index()

    fig = px.line(
        recent_df,
        x="DateTime",
        y="Electricity_Consumption",
        markers=True,
        title="Recent Electricity Consumption"
    )

    fig.update_layout(
        xaxis_title="DateTime",
        yaxis_title="Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "This forecasting page demonstrates the model using "
        "historical data. A production system would connect "
        "the same workflow to continuously arriving measurements."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header("🤖 Model Evaluation")

    st.subheader("Model Comparison")

    display_results = model_performance.copy()

    display_results["MAE"] = display_results["MAE"].round(4)
    display_results["RMSE"] = display_results["RMSE"].round(4)
    display_results["R2"] = display_results["R2"].round(4)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        model_performance,
        x="Model",
        y="MAE",
        title="Model Comparison - MAE"
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="MAE"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Actual vs Predicted")

    plot_df = prediction_dashboard.copy()

    fig = px.line(
        plot_df.head(1000),
        x="DateTime",
        y=["Actual", "Predicted"],
        title="Actual vs Predicted Electricity Consumption"
    )

    fig.update_layout(
        xaxis_title="DateTime",
        yaxis_title="Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Prediction Error Distribution")

    error_fig = px.histogram(
        prediction_dashboard,
        x="Error",
        nbins=50,
        title="Prediction Error Distribution"
    )

    error_fig.update_layout(
        xaxis_title="Error (Actual - Predicted)",
        yaxis_title="Frequency"
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Model",
        kpis["Best Model"]
    )

    col2.metric(
        "MAE",
        f'{kpis["Best Model MAE"]:.4f}'
    )

    col3.metric(
        "RMSE",
        f'{kpis["Best Model RMSE"]:.4f}'
    )

    st.success(
        f'The {kpis["Best Model"]} model improves MAE by '
        f'{kpis["MAE Improvement Percent"]:.2f}% compared '
        "with the previous-hour baseline."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Electricity Consumption Analysis | "
    "Python • Pandas • Scikit-learn • Streamlit • Plotly"
)
