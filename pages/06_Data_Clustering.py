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

st.title("Data Clustering (K-Means)")

st.markdown(
    """
    In this section, we will create an unsupervised classification model using our custom **K-Means** algorithm. 
    
    Our goal is to group video games based on their sales performance in different regions to see if natural tiers or categories emerge.
    """
)

st.divider()

## ------------- Section 1: Initial Distribution ----------------
st.subheader("1. North America vs. Europe Sales")

st.markdown(
    """
    With the following chart, we can observe how North American sales relate to European sales. 
    
    You'll notice a massive concentration of games clustered near the bottom left (low sales in both regions), with a few outliers stretching into the higher numbers. Let's see if K-Means can successfully segment them!
    """
)

# Initial Scatter Plot
fig_initial = px.scatter(
    df, 
    x="NA_Sales", 
    y="EU_Sales", 
    color_discrete_sequence=['red'],
    title="Raw Data: NA Sales vs. EU Sales",
    labels={
        "NA_Sales": "North American Sales (Millions)",
        "EU_Sales": "European Sales (Millions)"
    },
    opacity=0.6,
    hover_data=["Name"]
)
st.plotly_chart(fig_initial, width='stretch')

st.divider()

## ------------- Section 2: Custom K-Means Implementation ----------------
st.subheader("2. Interactive K-Means Algorithm")

st.markdown(
    """
    Use the slider below to select the number of clusters (K). The algorithm will randomly initialize centroids, calculate the Euclidean distance for every point, assign clusters, and recalculate the centroids until the model converges.
    """
)

# User input for K
user_k = st.slider("Select the number of clusters (K)", min_value=2, max_value=7, value=2, step=1)

# --- Optimized Custom K-Means using NumPy ---
def run_custom_kmeans(X, k, max_iters=100, tol=1e-2):
    """
    Vectorized implementation of the custom K-Means algorithm for Streamlit performance.
    """
    np.random.seed(42) # For reproducibility
    
    # 1. Initialize centroids randomly from the data points
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices]
    
    iterations = 0
    for i in range(max_iters):
        iterations += 1
        
        # 2. Assign points to the closest centroid using Euclidean distance
        # np.linalg.norm calculates the distance for all points instantly
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 3. Calculate new centroids (mean of the points in each cluster)
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] 
            for j in range(k)
        ])
        
        # 4. Check for convergence (if centroids barely moved)
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
            
        centroids = new_centroids
        
    return centroids, labels, iterations


if st.button("Run K-Means", type="primary"):
    
    # Prepare the data as a NumPy array (x=NA_Sales, y=EU_Sales)
    X_data = df[['NA_Sales', 'EU_Sales']].values
    
    # Run our vectorized custom algorithm
    final_centroids, final_labels, total_iters = run_custom_kmeans(X_data, user_k)
    
    # Create a copy of the dataframe to store the cluster labels for plotting
    plot_df = df.copy()
    plot_df['Cluster'] = [f"Cluster {label + 1}" for label in final_labels]
    
    st.success(f"Model converged in **{total_iters}** iterations!")
    
    # --- Plot the Results ---
    fig_clusters = px.scatter(
        plot_df, 
        x="NA_Sales", 
        y="EU_Sales", 
        color="Cluster",
        title=f"K-Means Clustering Results (K={user_k})",
        labels={
            "NA_Sales": "North American Sales (Millions)",
            "EU_Sales": "European Sales (Millions)"
        },
        hover_data=["Name"],
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Add the Centroids as large black X marks
    fig_clusters.add_trace(go.Scatter(
        x=final_centroids[:, 0],
        y=final_centroids[:, 1],
        mode='markers',
        name='Centroids',
        marker=dict(color='white', size=12, symbol='x'),
        hoverinfo='skip'
    ))
    
    st.plotly_chart(fig_clusters, width='stretch')
    
    # --- Dynamic Conclusion ---
    if user_k == 2:
        st.info(
            """
            **💡 Conclusion (K=2):** 
            As we can appreciate in the chart, K-Means successfully separated the data into two distinct sets. 
            The upper/right cluster represents the blockbusters—games that sold incredibly well in both America and Europe. The larger, lower-left cluster represents the vast majority of games that had average or low sales across both regions.
            """
        )