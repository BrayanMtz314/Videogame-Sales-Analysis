import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# importing the data
df = pd.read_csv("data/vgsales.csv")
df = df.convert_dtypes()

st.title("Data Cleaning")

st.markdown(
    """
    In this section, we will explore the dataset, check for null and duplicate values, and discard unnecessary data. This is an essential step in preparing the dataset for further analysis and modeling.
    """
)

st.divider()

st.header("Initial exploratory")
st.markdown(
    """
    We will use pandas to work with the data at the various stages of the process. The built-in pandas function `info()` can be used to see how many null values ​​the dataset contains.
    """
)

df_info = pd.DataFrame(
    {
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values
    }
)

st.dataframe(df_info)


st.divider()
st.header("Data Cleaning Steps")

st.write("For this dataset, we don't want to keep the 'Rank' column, as it is not necessary for our analysis. We will drop this column from the dataset.")

df.drop(columns=["Rank"], inplace=True)

st.dataframe(df.head())

st.markdown("Also we can check for duplicate values in the dataset. We will use the `duplicated()` function to find any duplicate rows.")

st.write("Number of duplicate rows:", df.duplicated().sum())

st.markdown("Another thing that we don't want is to keep null values in any of the columns. We will use the `dropna()` function to remove any rows with null values.")

st.write("Number of rows before dropping null values:", df.shape[0])

df.dropna(inplace=True)

st.write("Number of rows after dropping null values:", df.shape[0])

st.divider()
st.header("Final Dataset")

st.dataframe(df.head())

df.to_csv("data/vgsales_cleaned.csv", index=False)

