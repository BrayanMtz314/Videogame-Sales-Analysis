import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Descriptive Analysis")

# Cache the data loading to make the app much faster
@st.cache_data
def load_data():
    return pd.read_csv("data/vgsales_cleaned.csv")

df = load_data()

# --- Functions ---
def descriptive_data(df):
    """Returns descriptive statistics for the numerical columns."""
    return df[["Global_Sales", "Other_Sales", "JP_Sales", "NA_Sales", "EU_Sales"]].agg(["mean", "median", "sum", "var", "std", "kurt"])

def top10_games(df):
    """Returns the top 10 best-selling games."""
    return df.sort_values(by="Global_Sales", ascending=False).head(10)[["Name", "Global_Sales"]]

def total_by_genre(df):
    """Returns the total global sales grouped by genre."""
    tabla = df.pivot_table(
        values=["Global_Sales"],
        index="Genre",
        aggfunc="sum"
    ).reset_index()
    return tabla.sort_values(by="Global_Sales", ascending=False)

def total_by_year(df):
    """Returns the total global sales grouped by year."""
    table = df.pivot_table(
        values=["Global_Sales"],
        index="Year",
        aggfunc="sum"
    ).reset_index()
    return table

def top_publishers(df, n=5):
    """Calculates top N publishers by global sales and plots a pie chart."""
    ventas_pub = df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
    top_n = ventas_pub.head(n)
    otros = ventas_pub.iloc[n:].sum()
    
    tabla = pd.concat([top_n, pd.Series({"Others": otros})])
    tabla = tabla.reset_index()
    tabla.columns = ["Publisher", "Global_Sales"]
    
    total = tabla["Global_Sales"].sum()
    tabla["Porcentaje"] = (tabla["Global_Sales"] / total) * 100

    # FIX: Use fig, ax for Streamlit to control size and ensure thread safety
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Adding a nice color palette and a slight shadow for aesthetics
    ax.pie(
        tabla["Global_Sales"], 
        labels=tabla["Publisher"], 
        autopct="%1.1f%%", 
        startangle=140,
        colors=plt.cm.Set3.colors,
        wedgeprops={'edgecolor': 'white'}
    )
    ax.set_title("Top Publishers of Video Games (Global Sales)", fontsize=14, pad=20)
    
    # Render in Streamlit
    st.pyplot(fig)


# --- Load Data Variables ---
desc_data = descriptive_data(df)
top10 = top10_games(df)
totalByGenre = total_by_genre(df)
table = total_by_year(df)


# --- Dashboard Layout ---

st.markdown(
    """
    Welcome to the descriptive analysis stage! Here, we explore how sales are distributed across regions and uncover the top-performing games, genres, and publishers. Visualizing these metrics helps us better understand the true nature of our data.
    """
)

st.divider()    

## ------------- Section 1 ----------------
st.header("1. Descriptive Data")

st.markdown(
    """
    This section provides a statistical overview of the numerical data, including measures of central tendency (like the mean and median) and dispersion (such as standard deviation).
    """
)

st.dataframe(desc_data, width='stretch')

st.info(
    """
    **Key Takeaways:**
    - North America accounts for nearly half of global video game sales historically, followed by the European and Japanese markets.
    """
)

st.divider()

## ------------- Section 2 ----------------
st.header("2. Top Best-Selling Games")

st.markdown(
    """
    The chart below highlights the top 10 best-selling games worldwide based on the dataset.
    """
)

st.bar_chart(
    top10.sort_values(by='Global_Sales', ascending=False), 
    x="Name", 
    y="Global_Sales",
    color="#8A2BE2"  # Violet hex code
)

st.info(
    """
    **Key Takeaways:**
    - *Wii Sports* is the highest-selling video game in this dataset. Note that this data only covers sales up to roughly 2015; current figures in 2026 include more recent blockbusters that may have surpassed these numbers.
    """
)

st.divider()

## ------------- Section 3 ----------------
st.header("3. Total Revenue by Genre")

st.markdown(
    """
    The following bar chart illustrates the most profitable video game genres based on historical global sales.
    """
)

st.bar_chart(
    totalByGenre.sort_values(by='Global_Sales', ascending=False), 
    x="Genre", 
    y="Global_Sales",
    horizontal=True,
    color="#2E8B57"  # SeaGreen hex code
)

st.info(
    """
    **Key Takeaways:**
    - The top three highest-grossing genres are Action, Sports, and Shooters.
    """
)

st.divider()

## ------------- Section 4 ----------------
st.header("4. Total Sales by Year")

st.markdown(
    """
    Tracking total revenue by year reveals industry trends and highlights the most profitable years in gaming history.
    """
)

st.dataframe(table, width='stretch')

st.info(
    """
    **Key Takeaways:**
    - The year with the highest revenue was 2008, with $678 million in global sales, followed closely by 2009 and 2007.
    - The data shows a decline in recorded revenue after 2010. Because the gaming industry has grown massively in recent years, this confirms the dataset is capped around the mid-2010s.
    """
)

st.divider()

## ------------- Section 5 ----------------
st.header("5. Top Publishers")

st.markdown(
    """
    Finally, we break down the top 5 publishers by global revenue. All other publishers are grouped into a single "Others" category to show their combined market share.
    """
)

# Call the function to display the pie chart
top_publishers(df) 

st.info(
    """
    **Key Takeaways:**
    - While the combined "Others" category holds the largest market share overall, Nintendo and Electronic Arts dominate as the top individual publishers.
    """
)