import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    df = df.sample(1000) # The dataset is large, so I use sampled subsets for efficient processing and testing

    # Basic Information
    st.subheader("Basic Information")
    st.write("Shape of the dataset:", df.shape)
    st.write("Summary Statistics:")
    st.write(df.describe())
    st.write("Columns and Data Types:")
    st.write(df.dtypes)

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Interactive Visualizations
    st.subheader("Interactive Visualizations")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_x = st.selectbox("Select X-axis:", numeric_cols)
    selected_y = st.selectbox("Select Y-axis:", numeric_cols)

    if selected_x and selected_y:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=selected_x, y=selected_y, ax=ax)
        st.pyplot(fig)

    # Categorical Analysis
    st.subheader("Categorical Analysis")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    selected_cat = st.selectbox("Select a categorical column:", categorical_cols)

    if selected_cat:
        st.write(df[selected_cat].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=selected_cat, ax=ax)
        st.pyplot(fig)
