import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading
@st.cache_data
def load_data():
    return pd.read_csv("data/vgsales_cleaned.csv")

df = load_data()

# Group data by year. (Note: using sum() gives us total sales, not the average)
total_years = df.groupby("Year")["Global_Sales"].sum().reset_index()

st.title("Time Series Forecasting")

st.markdown(
    """
    In this section, we will create a forecasting model using **Linear Regression**. 
    
    We will treat the **Year** as our independent variable (X) and the **Total Global Sales** as our dependent variable (y) to predict future market trends.
    """
)

st.divider()

## ------------- Section 1: Time Series Distribution ----------------
st.subheader("1. Global Sales Over Time")

st.markdown("First, let's look at the distribution of total global sales by year to understand the historical trend.")

# Line chart showing the time series
fig_ts = px.line(
    total_years, 
    x="Year", 
    y="Global_Sales", 
    markers=True,
    title="Total Global Sales per Year",
    labels={"Global_Sales": "Total Global Sales (Millions)", "Year": "Release Year"}
)
st.plotly_chart(fig_ts, use_container_width=True)


st.divider()

## ------------- Section 2: Model Evaluation (All Data) ----------------
st.subheader("2. Initial Linear Regression Model")

st.markdown(
    """
    We will start by training our model on the entire dataset. 
    """
)

def build_regression_plot(X, y, title):
    """Helper function to train the model, calculate R2, and build a Plotly figure."""
    X_reshaped = np.array(X).reshape(-1, 1)
    y_array = np.array(y)

    # Train model
    modelo = LinearRegression()
    modelo.fit(X_reshaped, y_array)
    y_pred = modelo.predict(X_reshaped)
    r2 = r2_score(y_array, y_pred)

    # Build Plotly Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_array, mode='markers', name='Actual Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Regression Line', line=dict(color='red', width=2)))
    fig.update_layout(title=title, xaxis_title='Year (X)', yaxis_title='Global Sales (y)')

    return modelo, r2, fig

# Use all data
X_all = total_years["Year"]
y_all = total_years["Global_Sales"]
modelo_all, r2_all, fig_all = build_regression_plot(X_all, y_all, 'Regression on All Years')

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Initial R² Score", f"{r2_all:.4f}")
    st.error("The model **cannot** successfully predict the dependent variable.")
    st.markdown(
        """
        As we can see, the model fails to capture the trend. This is due to a sudden "break" or drop-off in the data after the year 2010. 
        
        This drop-off isn't necessarily a real-world market crash; it indicates an issue with the dataset's origin, where an equal volume of games was simply not recorded after this year.
        """
    )
with col2:
    st.plotly_chart(fig_all, use_container_width=True)

st.divider()

## ------------- Section 3: Model Refinement ----------------
st.subheader("3. Refined Model (1980 - 2010)")

st.markdown(
    """
    To fix the data quality issue, we will restrict the data exposed to the model, training it only on the video games released between **1980 and 2010**.
    """
)

# Filter data between 1980 and 2010
filtered_years = total_years[(total_years["Year"] >= 1980) & (total_years["Year"] <= 2010)]
X_filtered = filtered_years["Year"]
y_filtered = filtered_years["Global_Sales"]

modelo_filtered, r2_filtered, fig_filtered = build_regression_plot(X_filtered, y_filtered, 'Regression (1980 - 2010)')

col3, col4 = st.columns([1, 2])
with col3:
    st.metric("Refined R² Score", f"{r2_filtered:.4f}")
    st.success("The model **successfully** predicts the dependent variable!")
    st.markdown("Now that the model fits the reliable historical data, we can use it to predict sales for years beyond 2010 as if the data recording hadn't dropped off.")
with col4:
    st.plotly_chart(fig_filtered, use_container_width=True)

st.divider()

## ------------- Section 4: Interactive Forecasting ----------------
st.subheader("4. Predict Future Data")

st.markdown("Use the tools below to forecast global sales for future years using our refined model.")

col_single, col_multi = st.columns(2)

with col_single:
    st.markdown("**Predict a Single Year**")
    target_year = st.number_input("Enter a year to predict:", min_value=2011, max_value=2030, value=2012, step=1)
    
    if st.button("Predict Single Year", type="primary"):
        pred = modelo_filtered.predict([[target_year]])
        st.info(f"📈 For the year **{target_year}**, the model predicts **{pred[0]:.2f} million** in global sales.")

with col_multi:
    st.markdown("**Predict a Range of Years**")
    year_range = st.slider("Select a range of years:", min_value=2011, max_value=2030, value=(2011, 2015))
    
    if st.button("Predict Range", type="primary"):
        # Generate the list of years
        years_list = list(range(year_range[0], year_range[1] + 1))
        years_array = np.array(years_list).reshape(-1, 1)
        
        # Make predictions
        predictions = modelo_filtered.predict(years_array)
        
        # Display as a clean dataframe
        pred_df = pd.DataFrame({
            "Year": years_list,
            "Predicted Global Sales (Millions)": [f"{p:.2f}" for p in predictions]
        })
        st.dataframe(pred_df, hide_index=True)