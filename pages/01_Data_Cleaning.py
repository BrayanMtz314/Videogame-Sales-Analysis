import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading to make the app much faster
@st.cache_data
def load_data():
    df = pd.read_csv("data/vgsales.csv")
    return df.convert_dtypes()

# Import the data
df = load_data()

st.title("Data Cleaning")

st.markdown(
    """
    In this section, we will explore the raw dataset, check for missing or duplicate values, and discard unnecessary columns. 
    This is an essential preprocessing step to ensure our data is accurate and ready for analysis and hypothesis testing.
    """
)

st.divider()

## ------------- Section 1 ----------------
st.header("1. Initial Data Exploration")
st.markdown(
    """
    First, let's look at the structure of our data. Using pandas, we can extract the data types and count the non-null values for each column, similar to what the `.info()` method provides in a notebook.
    """
)

# Create the info dataframe. Converting dtypes to string prevents Streamlit rendering errors.
df_info = pd.DataFrame(
    {
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values, 
        "Non-Null Count": df.notnull().sum().values
    }
)

st.dataframe(df_info, width='stretch')

st.divider()

## ------------- Section 2 ----------------
st.header("2. Data Cleaning Steps")

# --- Step A ---
st.subheader("Dropping Unnecessary Columns")
st.markdown("The `Rank` column acts as a secondary index and isn't necessary for our statistical analysis, so we will drop it from the dataset.")

# Assign to a new variable to avoid mutating the cached dataframe
cleaned_df = df.drop(columns=["Rank"])
st.dataframe(cleaned_df.head(), width='stretch')

# --- Step B ---
st.subheader("Handling Duplicates")
st.markdown("Next, we check for exact duplicate rows across the dataset.")

duplicates = cleaned_df.duplicated().sum()
if duplicates > 0:
    st.warning(f"Found **{duplicates}** duplicate rows. Removing them...")
    cleaned_df = cleaned_df.drop_duplicates()
else:
    st.success(f"Number of duplicate rows found: **{duplicates}**")

# --- Step C ---
st.subheader("Handling Missing Values")
st.markdown("To maintain data integrity for our statistical tests, we must deal with null values. For this analysis, we will drop any rows that contain missing data.")

# Use columns to show the "Before" and "After" clearly side-by-side
col1, col2 = st.columns(2)
col1.metric("Rows before dropping nulls", cleaned_df.shape[0])

cleaned_df = cleaned_df.dropna()

col2.metric("Rows after dropping nulls", cleaned_df.shape[0])


st.divider()

## ------------- Section 3 ----------------
st.header("3. Final Dataset")
st.markdown("Here is a preview of our fully cleaned dataset, ready for the next stages of our pipeline:")

st.dataframe(cleaned_df.head(), width='stretch')

# Instead of overwriting the local file on every app rerun, provide a download button!
csv = cleaned_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Cleaned Data as CSV",
    data=csv,
    file_name='vgsales_cleaned.csv',
    mime='text/csv',
    type="primary"
)