import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Video Game Sales Analysis Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Video Game Sales Analysis Project")

st.markdown(
    """
    This dashboard presents a comprehensive analysis of the well-known 2016 Kaggle dataset, [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) by GregorySmith. 
    
    This dataset records global sales for various games and enables tasks such as data cleaning, exploratory analysis, regression, and classification, in addition to answering numerous questions about the video game industry.
    """
)
st.divider()

st.header("Dataset Columns")

st.markdown(
    """
    * **Rank** - Ranking of overall sales
    * **Name** - The game's name
    * **Platform** - Platform of the game's release (i.e., PC, PS4, etc.)
    * **Year** - Year of the game's release
    * **Genre** - Genre of the game
    * **Publisher** - Publisher of the game
    * **NA_Sales** - Sales in North America (in millions)
    * **EU_Sales** - Sales in Europe (in millions)
    * **JP_Sales** - Sales in Japan (in millions)
    * **Other_Sales** - Sales in the rest of the world (in millions)
    * **Global_Sales** - Total worldwide sales
    """
)

st.divider()

st.header("A quick look at the raw dataset")

# Cache the initial load so the home page is snappy
@st.cache_data
def load_preview():
    return pd.read_csv("data/vgsales.csv", nrows=10)

st.dataframe(load_preview(), width='stretch')

st.divider()

st.header("Project Content")
st.markdown("Use the sidebar to navigate through the different stages of this analysis pipeline:")
st.write("") # Add a little vertical space

# Using a list of dictionaries makes generating the UI much cleaner!
pages = [
    {
        "title": "1. Data Cleaning", 
        "desc": "On this page, we begin exploring the raw dataset, examine missing and duplicate values, and discard unnecessary data to prepare for our statistical models."
    },
    {
        "title": "2. Descriptive Analysis", 
        "desc": "Here, we perform a descriptive analysis of the dataset, including statistical summaries, top-selling games, revenue by genre, and market share by top publishers."
    },
    {
        "title": "3. Hypothesis Testing", 
        "desc": "We use the Shapiro-Wilk test for normality and the Kruskal-Wallis non-parametric test to determine mathematically if video game genres significantly impact global sales."
    },
    {
        "title": "4. Linear Regression Analysis", 
        "desc": "Explores the correlation between numerical variables and uses a Simple Linear Regression model to analyze the direct relationship between North American and Global sales."
    },
    {
        "title": "5. Data Classification (KNN)", 
        "desc": "An interactive K-Nearest Neighbors model that predicts the console (Platform) of a theoretical game based entirely on its North American and Japanese sales coordinates."
    },
    {
        "title": "6. Data Clustering (K-Means)", 
        "desc": "Applies a custom, fully vectorized K-Means clustering algorithm to group video games into performance tiers based on their sales in North America and Europe."
    },
    {
        "title": "7. Forecasting", 
        "desc": "Uses Linear Regression on historical time series data (1980-2010) to forecast expected future global sales and demonstrate how to handle data volume drop-offs."
    }
]   

# Loop through the pages to generate the columns automatically
for page in pages:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(page["title"])
    with col2:
        st.write(page["desc"])
    st.divider()