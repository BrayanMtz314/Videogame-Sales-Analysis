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

def top10_games(df):
    """Return the top 10 best selling games"""
    top10 = df.sort_values(by="Global_Sales", ascending=False).head(10)[["Name", "Global_Sales"]]
    return top10

def total_by_genre(df):
    """Return the total by genre"""
    tabla = df.pivot_table(
        values=["Global_Sales"],
        index="Genre",
        aggfunc="sum").reset_index()
    return tabla.sort_values(by="Global_Sales", ascending=False)

def total_by_year(df):
    table = df.pivot_table(
        values=["Global_Sales"],
        index="Year",
        aggfunc="sum").reset_index()
    return table

    



desc_data = descriptive_data(df)
top10 = top10_games(df)
totalByGenre = total_by_genre(df)
table = total_by_year(df)


st.markdown(
    """
    In this stage, we move on to the "fun" part: we will look at how numerical data is distributed across different regions and analyze various "top" lists (such as the best games of the year, leading publishers, etc.).
    To do this, we will use different types of charts. This phase is very useful, as it allows us to visualize the true nature of the data.
    """
    )

st.divider()    


## ------------- Section 1 ----------------
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

st.divider()

## ------------- Section 2 ----------------

st.header("Top best-selling games")

st.markdown(
    """
    One piece of information we can derive from this dataset is which games are the best-selling worldwide;
    we can determine this from the following table:
    """
)

st.bar_chart(
    top10.sort_values(by='Global_Sales', ascending=False), 
    x="Name", 
    y="Global_Sales"
)

st.subheader("Interest things that we can recover of this section")

st.markdown(
    """
    - Wii Sports is the highest-selling video game in this dataset; keep in mind that these are 2015 data. As of today, in 2026, many games have recorded higher sales.
    """
    )
st.divider()

## ------------- Section 3 ----------------

st.header("Total by genre")

st.markdown(
    """
    We can also determine how much a genre grosses across the dataset; this bar chart shows the most profitable ones.
    """
)

st.bar_chart(
    totalByGenre.sort_values(by='Global_Sales', ascending=False), 
    x="Genre", 
    y="Global_Sales",
    horizontal=True
)

st.subheader("Interest things that we can recover of this section")

st.markdown(
    """
    - The three genres with the highest revenue are Action, Sport and Shooter.
    """
    )
st.divider()

## ------------- Section 4 ----------------

st.header("Total by Year")

st.markdown(
    """
    We can also determine how much a genre grosses across the dataset; this bar chart shows the most profitable ones.
    """
)

st.dataframe(table)

st.subheader("Interest things that we can recover of this section")

st.markdown(
    """
    - The three genres with the highest revenue are Action, Sport and Shooter.
    """
    )
st.divider()

