import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Descriptive Analysis")

df = pd.read_csv("data/vgsales_cleaned.csv")

# functios
def descriptive_data(df):
    """This function returns descriptive statistics for the numerical columns in the dataset."""
    return df[["Global_Sales", "Other_Sales", "JP_Sales","NA_Sales", "EU_Sales"]].agg(["mean", "median", "sum", "var", "std", "kurt"])

desc_data = descriptive_data(df)

st.markdown(
    """
    In this stage, we move on to the "fun" part: we will look at how numerical data is distributed across different regions and analyze various "top" lists (such as the best games of the year, leading publishers, etc.).
    To do this, we will use different types of charts. This phase is very useful, as it allows us to visualize the true nature of the data.
    """
    )

st.divider()

st.header("Descriptive Data")

st.markdown(
    """
    In this section, we can see the descriptive statistics of the numerical columns within the dataset, we can see the mean, median, standard deviation, and other measures of central tendency and dispersion.
    """
)

st.dataframe(desc_data)

st.subheader("Interest things that we can recover of this section")

st.markdown(
    """
    - North America has accounted for half of global video game sales over the years, followed by the markets in Europe and Japan.
    """
    )