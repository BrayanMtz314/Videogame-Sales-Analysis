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
    This is a comprehensive analysis of the well-known 2016 Kaggle dataset [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) by GregorySmith. This dataset records global sales 
    or various games and enables tasks such as data cleaning, exploratory analysis, regression, and classification, in addition to answering numerous 
    questions about video game sales.
    """)
st.divider()

st.header("Columns")


st.markdown(
    """
    - Rank - Ranking of overall sales
    - Name - The games name
    - Platform - Platform of the games release (i.e. PC,PS4, etc.)
    - Year - Year of the game's release
    - Genre - Genre of the game
    - Publisher - Publisher of the game
    - NA_Sales - Sales in North America (in millions)
    - EU_Sales - Sales in Europe (in millions)
    - JP_Sales - Sales in Japan (in millions)
    - Other_Sales - Sales in the rest of the world (in millions)
    - Global_Sales - Total worldwide sales.
    """
    )

st.divider()
st.header("A little look at the dataset")

df = pd.read_csv("data/vgsales.csv", nrows=10)

st.dataframe(df)

st.divider()

st.header("Content")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Data Cleaning")

with col2:
    st.write("On this page, we begin exploring the dataset, examine null and duplicate values, and discard unnecessary data.")

st.divider()

col3, col4 = st.columns([1,2])

with col3:
    st.subheader("Descriptive Analysis")
with col4:
    st.write("In this section, we will perform a descriptive analysis of the dataset, including visualizations and statistical summaries.")
