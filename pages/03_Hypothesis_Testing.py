import streamlit as st
import pandas as pd
from scipy import stats

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading to make the app much faster
@st.cache_data
def load_data():
    return pd.read_csv("data/vgsales_cleaned.csv")

df = load_data()

st.title("Hypothesis Testing")

st.markdown(
    """
    Hypothesis testing is a statistical method that allows us to draw conclusions about an entire population based on sample data. It involves formulating two competing hypotheses: 
    * **The Null Hypothesis (H0):** Represents a statement of "no effect" or "no difference."
    * **The Alternative Hypothesis (H1):** Represents a statement of an actual effect or difference.
    
    Our goal here is to determine whether global sales differ significantly based on a video game's **genre**. To do this properly, we first need to check if the sales data for each genre is normally distributed. This will dictate whether we use a parametric or non-parametric test.
    """
)

st.divider()

## ------------- Section 1 ----------------
st.subheader("1. Normality Test (Shapiro-Wilk)")

st.markdown(
    """
    To assess normality, we use the **Shapiro-Wilk test**. 
    
    * **H0:** The data follows a normal distribution.
    * **H1:** The data does *not* follow a normal distribution.
    
    We evaluate this using a standard significance level ($\alpha = 0.05$). If our resulting p-value is less than 0.05, we reject H0 and conclude the data is not normally distributed. Let's run this test for every genre in our dataset:
    """
)

results = []

for genre, group in df.groupby("Genre"):
    values = group["Global_Sales"].values
    
    if len(values) >= 3:  
        stat, p_value = stats.shapiro(values)
        
        # Determine the normality status based on the p-value
        if p_value < 0.05:
            status = "Not Normal"
        else:
            status = "Normal"
            
        # Append the results as a dictionary to our list
        results.append({
            "Genre": genre,
            "P-Value": p_value,
            "Distribution": status
        })

shapiro_df = pd.DataFrame(results)
shapiro_df["P-Value"] = shapiro_df["P-Value"].apply(lambda x: f"{x:.2e}")

st.dataframe(shapiro_df, width='stretch')

st.warning(
    """
    **Insight:** As we can see, **none** of the genres exhibit a normal distribution. 
    
    Because of this, we cannot use standard parametric tests (like ANOVA). Instead, we must use a non-parametric alternative that does not rely on normally distributed data. For this exercise, we will use the **Kruskal-Wallis test**.
    """
)

st.divider()

## ------------- Section 2 ----------------
st.subheader("2. Kruskal-Wallis Test")

st.markdown(
    """
    The Kruskal-Wallis test is a non-parametric statistical method used to determine if there are significant differences between two or more independent groups. 
    
    * **H0:** There are no significant differences in median global sales between the genres.
    * **H1:** At least one genre differs significantly from the others.
    
    Again, we use a significance level of 0.05.
    """
)

# Extract arrays of global sales for each genre
groups = [group["Global_Sales"].values for name, group in df.groupby("Genre")]

# Apply Kruskal-Wallis
h_stat, p_value = stats.kruskal(*groups)

# Display the raw metrics side-by-side
col1, col2 = st.columns(2)
col1.metric("H-Statistic", f"{h_stat:.2f}")
col2.metric("P-Value", f"{p_value:.2e}")

# Display a clean interpretation based on the p-value
if p_value < 0.05:
    st.error(
        "**Result: Reject H0** \n\n"
        "The p-value is practically zero (< 0.05). There are significant differences between at least one pair of genres."
    )
else:
    st.success(
        "**Result: Fail to reject H0** \n\n"
        "The p-value is >= 0.05. There is insufficient evidence to suggest that the genres differ in global sales."
    )
    
st.divider()    
    
## ------------- Conclusion ----------------
st.subheader("Conclusion")

st.markdown(
    """
    Based on the highly significant results of the Kruskal-Wallis test, we can confidently conclude that **global sales vary significantly depending on the genre of the video game**. 
    
    This proves mathematically what we saw visually in the descriptive analysis: the genre of a video game has a substantial impact on its overall sales performance.
    """
)