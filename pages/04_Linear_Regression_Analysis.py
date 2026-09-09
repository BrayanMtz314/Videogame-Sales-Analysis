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
    # Assuming you saved the cleaned file from the previous step here
    return pd.read_csv("data/vgsales_cleaned.csv")

df = load_data()

st.title("Linear Regression Analysis")

st.markdown(
    """
    To begin our practice, we look for variables that have a high degree of correlation. To do this, we created the following heatmap of the correlation matrix. 
    
    Each cell in the matrix indicates the level of correlation between the row and column variables. The closer the value is to -1 or 1, the greater the degree of correlation.
    """
)

## ------------- Section 1: Correlation Matrix ----------------
st.subheader("Correlation Matrix")

# Calculate correlation only on numeric columns
numeric_df = df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()

# Create interactive heatmap using Plotly Express
fig_corr = px.imshow(
    corr_matrix, 
    text_auto=".2f", 
    color_continuous_scale="RdBu_r",
    aspect="auto",
)
st.plotly_chart(fig_corr, width='stretch')

st.divider()

## ------------- Section 2: Linear Regression ----------------
st.subheader("Simple Linear Regression")

st.markdown(
    """
    Although it is an obvious choice, we chose to perform a linear regression using **North America Sales (NA_Sales)** and **Global Sales (Global_Sales)**, as they exhibit the highest degree of correlation. 
    
    The choice is obvious because North American sales directly contribute to global sales, but this serves perfectly to test the scikit-learn methods and generate our regression chart.
    """
)

# Define variables (NA_Sales as independent, Global_Sales as dependent)
X = df[['NA_Sales']] 
y = df['Global_Sales']

# Create and train the model
modelo = LinearRegression()
modelo.fit(X, y)

# Predictions and R2 Score
y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)

# Use columns to put the explanation on the left and the chart on the right
col1, col2 = st.columns([1, 2])

with col1:
    # Display the R2 Score prominently
    st.metric(label="R² Score", value=f"{r2:.4f}")
    
    if r2 > 0.80:
        st.success("The model successfully predicts the dependent variable.")
    else:
        st.error("The model does not successfully predict the dependent variable.")
        
    st.markdown(
        """
        A desirable $R^2$ Score is close to 1, indicating that the model perfectly fits most of the data along the regression line with high significance. Conversely, if the $R^2$ score approaches 0, it indicates that the model fails to predict most of the data.
        
        **Note on the Outlier:**
        You might notice a data point positioned extremely high on the graph. This has been reviewed, and it is not an outlier or an error. It represents the sales for **Wii Sports**, which, in this specific dataset, surpasses the second-place game by almost double, making it by far the best-selling game recorded.
        """
    )

with col2:
    # Create the interactive scatter plot and regression line using Plotly Graph Objects
    fig_reg = go.Figure()

    # Scatter plot for the actual data
    fig_reg.add_trace(go.Scatter(
        x=df['NA_Sales'], 
        y=df['Global_Sales'], 
        mode='markers', 
        name='Actual Data',
        marker=dict(color='blue', opacity=0.5),
        text=df['Name'], # This allows the game name to show up when hovering!
        hovertemplate="<b>%{text}</b><br>NA Sales: %{x}M<br>Global Sales: %{y}M<extra></extra>"
    ))

    # Line plot for the regression line
    fig_reg.add_trace(go.Scatter(
        x=df['NA_Sales'], 
        y=y_pred, 
        mode='lines', 
        name='Regression Line',
        line=dict(color='red', width=2)
    ))

    # Format the chart layout
    fig_reg.update_layout(
        title='Simple Linear Regression: NA Sales vs. Global Sales',
        xaxis_title='Independent Variable (NA_Sales)',
        yaxis_title='Dependent Variable (Global_Sales)',
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig_reg, width='stretch')