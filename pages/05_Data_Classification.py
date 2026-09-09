import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading
@st.cache_data
def load_data():
    return pd.read_csv("data/vgsales_cleaned.csv")

df = load_data()

st.title("Data Classification (K-Nearest Neighbors)")

st.markdown(
    """
    This section utilizes the **K-Nearest Neighbors (KNN)** algorithm to predict the classification of a new data point. 
    
    In this example, we will use it to predict which console (`Platform`) a game belongs to, solely based on its North American and Japanese sales. The sales are represented as an (x, y) coordinate point, and the algorithm measures the Euclidean distance to find the closest games:
    
    $$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
    """
)

st.divider()

## ------------- Section 1: Initial Distribution ----------------
st.subheader("Console Distribution by Sales")
st.markdown("This scatter plot serves as a visual aid to see how games are distributed based on their North American (x) and Japanese (y) sales according to their console.")

# Create the initial distribution plot with Plotly
fig_dist = px.scatter(
    df, 
    x="NA_Sales", 
    y="JP_Sales", 
    color="Platform",
    title="Platform Distribution based on Sales",
    labels={
        "NA_Sales": "North American Sales (Millions)",
        "JP_Sales": "Japanese Sales (Millions)"
    },
    opacity=0.7,
    hover_data=["Name"] # Shows the game name on hover!
)
st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

## ------------- Section 2: Interactive KNN Model ----------------
st.subheader("Test the KNN Algorithm")
st.markdown("Enter the sales parameters below to plot a new game. The algorithm will find the $k$ nearest neighbors and predict which platform this theoretical game belongs to.")

# --- UI for User Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    user_na = st.number_input("North American Sales (x)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)
with col2:
    user_jp = st.number_input("Japanese Sales (y)", min_value=0.0, max_value=50.0, value=8.0, step=1.0)
with col3:
    user_k = st.slider("Number of Neighbors (k)", min_value=1, max_value=15, value=3, step=2)

# --- Optimized KNN Algorithm ---
def run_knn(data, na_val, jp_val, k):
    # Vectorized Euclidean Distance (much faster than looping through data.loc)
    distances = np.sqrt((data['NA_Sales'] - na_val)**2 + (data['JP_Sales'] - jp_val)**2)
    
    # Get the indices of the 'k' smallest distances
    nearest_indices = distances.nsmallest(k).index
    
    # Retrieve the platforms of these nearest neighbors
    nearest_platforms = data.loc[nearest_indices, 'Platform']
    
    # The prediction is the most frequent platform among the neighbors (the mode)
    prediction = nearest_platforms.mode()[0]
    
    return prediction, nearest_platforms.tolist()


# Run button to trigger the calculation
if st.button("Predict Platform", type="primary"):
    
    # Run the model
    prediction, neighbors = run_knn(df, user_na, user_jp, user_k)
    
    st.success(f"### Predicted Platform: **{prediction}**")
    st.info(f"The {user_k} nearest neighbors belonged to these platforms: {', '.join(neighbors)}")
    
    # --- Plot the results with the new point ---
    fig_result = px.scatter(
        df, 
        x="NA_Sales", 
        y="JP_Sales", 
        color="Platform",
        labels={
            "NA_Sales": "North American Sales (Millions)",
            "JP_Sales": "Japanese Sales (Millions)"
        },
        opacity=0.4 # Make background points more transparent
    )
    
    # Add the user's new point as a giant black star
    fig_result.add_trace(go.Scatter(
        x=[user_na],
        y=[user_jp],
        mode='markers+text',
        name='NEW GAME',
        marker=dict(color='white', size=15, symbol='star'),
        text=["NEW GAME"],
        textposition="top center",
        textfont=dict(color='white', size=12, family='Arial Black')
    ))
    
    fig_result.update_layout(title='KNN Prediction Result')
    st.plotly_chart(fig_result, use_container_width=True)