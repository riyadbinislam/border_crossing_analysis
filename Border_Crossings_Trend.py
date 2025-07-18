import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import streamlit as st
from sklearn.metrics import silhouette_score, davies_bouldin_score
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings("ignore")


st.markdown(
    """
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 10px;
    }
    .stText {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    data = pd.read_csv("Border_Crossing_Entry_Data.csv")
    return data

st.title("📊 Border Crossing Data Analysis")

tab1, tab2, tab3 = st.tabs(
    ["Data Description", "Exploratory Data Analysis (EDA)", "Machine Learning"])


with tab1:
    st.header(f"{chr(128221)} Data Inspection and Description")
    data = load_data()

    st.subheader(f"{chr(128269)} Data Overview")
    st.dataframe(data.head(10))
    st.dataframe(data.tail(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{chr(128207)} Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        st.subheader(f"{chr(10071)} Check for Duplicates")
        st.write(f"Unique Rows: {data.nunique().sum()} out of {data.shape[0]}")

        st.subheader(f"{chr(128204)} Missing Values")
        st.write(data.isnull().sum())

    with col2:
        st.subheader(f"{chr(128202)} Summary Statistics")
        st.write(data.describe())

    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"{chr(128194)} Data Information")
        info_df = pd.DataFrame({
            "Column": data.columns,
            "Non-Null Count": data.notnull().sum().values,
            "Data Type": data.dtypes.values,
        })
        st.dataframe(info_df)

    with col4:
        st.subheader(f"{chr(128209)} Unique Instances")
        st.dataframe(data.nunique())


    st.subheader(f"{chr(128290)} Data Types in Columns")
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    num_cols = data.select_dtypes(include=np.number).columns.tolist()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"{chr(128736)} Categorical Variables")
        st.write(cat_cols)

    with col4:
        st.subheader(f"{chr(128290)}🔢 Numerical Variables")
        st.write(num_cols)

    st.success(f"{chr(9989)} Data Inspection Completed!")

with tab2:

    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Plotting Window")

    plot_options = [
        "",
        "1. Top 15 Most Common Port of Entry",
        "2. States of Border Crossings",
        "3. Medium of Border Crossings",
        "4. Top 10 Ports by Number of Border Crossings",
        "5. Border-wise Traffic (US-Canada vs US-Mexico)",
        "6. Time Series Analysis: Monthly Trend",
        "7. Time Series Analysis: Yearly Trend",
        "8. Seasonal Trends in Border Crossings",
        "9. Categorized Measure Trend by Season",
        "10. Categorized Measure Trend by State",
        "11. Border Crossings HeatMap",
        "12. Seasonal Trend: State-wise Traffic",
        "13. Seasonal Trend: Measure-wise Traffic",
        "14. Monthly Trend: State-wise Traffic",
        "15. Monthly Trend: Measure-wise Traffic",
        "16. Yearly Trend: State-wise Traffic",
        "17. Yearly Trend: Measure-wise Traffic",
        "18. Vehicular Traffic vs Pedestrian Traffic",
        "19. State-wise Heatmap"
    ]

    selected_plot = st.selectbox("Select a plot to display:", plot_options)

    if selected_plot == "1. Top 15 Most Common Port of Entry":
        fig, ax = plt.subplots(figsize=(12, 6))
        data['Port Name'].value_counts().head(15).plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Top 15 Most Common Port of Entry', fontsize=16)
        ax.set_xlabel('Port Name', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xticklabels(data['Port Name'].value_counts().head(15).index, rotation=90, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.text("Based on the frequency of appearance in the dataset, the bar chart displays the top 15 most common ports of border crossings. Among all the ports listed, Eastport appears the most, significantly exceeding other ports. However, this does not necessarily imply the highest number of crossings. The remaining ports, such as Buffalo Niagra Falls, Nogales, Champlain Rouses Point, and International Falls, have relatively similar frequencies, with nearly 4000 occurrences, suggesting that data entries are fairly distributed except for Eastport, possibly due to reporting or recording patterns rather than actual traffic volume.")

    elif selected_plot == "2. States of Border Crossings":
        fig, ax = plt.subplots(figsize=(10, 6))
        data['State'].value_counts().plot(kind='bar', ax=ax, color='salmon')
        ax.set_title('State of Border Crossings', fontsize=16)
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.text("The bar chart represents the number of recorded border crossings by the U.S. states. The chart highlights the states with the most frequent instances in the dataset. North Dakota has the highest number of appearances, followed by Washington, Maine, Montana, and Texas. States like New Mexico, Idaho, and Michigan have significantly higher border crossing entries. The distribution suggests that states bordering Canada have notably higher crossing entries than states bordering New Mexico. Probable causes can be the number of border entry points or recording patterns.")

    elif selected_plot == "3. Medium of Border Crossings":
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Measure'].value_counts().plot(kind='bar', ax=ax, color='green')
        ax.set_title('Medium of Border Crossings', fontsize=16)
        ax.set_xlabel('Measure', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.text("The bar graph depicts the frequency of border crossings by transportation method. The majority of crossings were documented by personal automobiles and their passengers. The measurements closest to having the most entries are empty and loaded truck containers. The dataset's lowest incidence was in trains, rail containers (both loaded and empty), and train passengers. According to this distribution, the most common entry method is road-based crossings (trucks and personal cars) caused by everyday commutes, personal trips, and commercial trade. Rail-based crossings, on the other hand, are less common due to inadequate infrastructure or a decreased dependence on rail transportation for border crossing.")

    elif selected_plot == "4. Top 10 Ports by Number of Border Crossings":
        fig, ax = plt.subplots(figsize=(12, 6))
        top_ports = data.groupby('Port Name')['Value'].sum().nlargest(10)
        top_ports.sort_values().plot(kind='barh', ax=ax, color='purple')
        ax.set_title('Top 10 Ports by Number of Border Crossings', fontsize=16)
        ax.set_xlabel('Total Crossings', fontsize=12)
        ax.set_ylabel('Port Name', fontsize=12)
        st.pyplot(fig)

        st.text("The horizontal bar chart shows the top ten ports with the most border crossings. These ports are the busiest entrance sites, with San Ysidro ranking first with the most crossings overall and El Paso coming in second. Laredo, Hidalgo, and Calexico have a lot of border traffic. Ports, including Brownsville, Otay Mesa, Detroit, Nogales, and Buffalo Niagara Falls, on the other hand, have comparatively few but still significant crossing counts. Compared to northern U.S.-Canada crossings, the dominance of San Ysidro and El Paso indicates that U.S.-Mexico border crossings see higher traffic. The population density and commuter patterns close to these ports may cause this tendency.")

    elif selected_plot == "5. Border-wise Traffic (US-Canada vs US-Mexico)":
        fig, ax = plt.subplots(figsize=(8, 5))
        border_traffic = data.groupby('Border')['Value'].sum()
        border_traffic.plot(kind='bar', ax=ax, color=['darkblue', 'darkred'])
        ax.set_title('Border-wise Traffic (US-Canada vs US-Mexico)', fontsize=16)
        ax.set_ylabel('Total Crossings', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.text("The bar chart compares the number of border crossings between U.S.-Canada and U.S.-Mexico borders. The U.S.-Mexico border experiences significantly higher traffic, with over 8 billion crossings, whereas the U.S.-Canada border has approximately 3 billion crossings. This notable difference can be attributed to higher population density along the southern border, increased trade activity between the U.S. and Mexico, and frequent cross-border commutation.")

    elif selected_plot == "6. Time Series Analysis: Monthly Trend":
        data['Date'] = pd.to_datetime(data['Date'], format='%b %Y', errors='coerce')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.to_period('M')

        min_year, max_year = int(data['Year'].min()), int(data['Year'].max())
        selected_year = st.slider("Select Year", min_year, max_year, (min_year, max_year))

        filtered_data = data[(data['Year'] >= selected_year[0]) & (data['Year'] <= selected_year[1])]
        monthly_trend = filtered_data.groupby('Month')['Value'].sum()

        st.subheader("Monthly Trend of Border Crossings")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(monthly_trend.index.astype(str), monthly_trend.values, linestyle='-', marker='o', color='blue', label='Monthly Crossings')
        ax.set_title('Monthly Trend of Border Crossings', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Total Crossings', fontsize=12)
        ax.set_xticks([i for i in range(0, len(monthly_trend), 12)])
        ax.set_xticklabels([str(idx.year) for idx in monthly_trend.index[::12]], fontsize=10, rotation=90)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)


    elif selected_plot == "7. Time Series Analysis: Yearly Trend":
        data['Date'] = pd.to_datetime(data['Date'], format='%b %Y', errors='coerce')
        yearly_trend = data.groupby(data['Date'].dt.year)['Value'].sum()
        st.subheader("Yearly Trend of Border Crossings")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(yearly_trend.index, yearly_trend.values, linestyle='-', marker='o', color='green', label='Yearly Crossings')
        ax.set_title('Yearly Trend of Border Crossings', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Total Crossings', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

        st.text("The line graph displays the yearly trend of border crossings from 1995 to 2025, emphasizing significant fluctuations. Crossings increased steadily from 1995 to 2000, reaching a peak of over 5 billion. However, a gradual decline followed from 2000 to 2010, with the numbers decreasing yearly. The trend stabilized with minor fluctuations until 2019, before a sharp drop in 2020, reaching the lowest point on the graph. This was followed by a strong rebound in 2021, with crossings increasing again but showing variability in the following years. Overall, the graph exhibits changing border crossing patterns.")

    elif selected_plot == "8. Seasonal Trends in Border Crossings":
        data['Date'] = pd.to_datetime(data['Date'], format='%b %Y', errors='coerce')
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
        seasonal_trend = data.groupby('Season')['Value'].sum()
        st.subheader("Seasonal Trends in Border Crossings")
        fig, ax = plt.subplots(figsize=(12, 6))
        seasonal_trend.plot(kind='bar', ax=ax, color=['red', 'green', 'blue', 'orange'])
        ax.set_title("Seasonal Trends in Border Crossings", fontsize=16)
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Total Crossings", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.text("The bar chart depicts the seasonal trends in border crossings, comparing the total number of crossings in Fall, Spring, Summer, and Winter. The highest number of crossings is documented for Summer, followed closely by Fall and Spring, indicating increased travel and movement during warmer months. Winter had the lowest total crossings, possibly due to colder weather, indicating reduced movement. The graph shows that seasonal changes influence border traffic patterns, with peak movement occurring in the summer months.")


    elif selected_plot == "9. Categorized Measure Trend by Season":
        data['Date'] = pd.to_datetime(data['Date'], format='%b %Y', errors='coerce')
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
        measures = data['Measure'].unique()
        st.subheader("Select a Measure to View Seasonal Trends")
        selected_measure = st.selectbox("Measure", measures)
        if selected_measure:
            measure_season_trend = data[data['Measure'] == selected_measure].groupby('Season')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            measure_season_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Seasonal Trend for {selected_measure}", fontsize=16)
            ax.set_xlabel("Season", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    elif selected_plot == "10. Categorized Measure Trend by State":
        data['Date'] = pd.to_datetime(data['Date'], format='%b %Y', errors='coerce')
        measures = data['Measure'].unique()

        st.subheader("Select a Measure to View State-wise Trends")
        selected_measure = st.selectbox("Measure", measures)

        if selected_measure:
            measure_state_trend = data[data['Measure'] == selected_measure].groupby('State')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            measure_state_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"State-wise Trend for {selected_measure}", fontsize=16)
            ax.set_xlabel("State", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)


    elif selected_plot == "11. Border Crossings HeatMap":
        st.subheader("Border Crossings HeatMap")
        border_map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=5)
        heat_data = data[['Latitude', 'Longitude', 'Value']].values.tolist()
        HeatMap(heat_data).add_to(border_map)
        st_folium(border_map, width=700, height=500)

        st.text("The heatmap displays the density of border crossings along the U.S.-Mexico, U.S.-Canada, and Alaska borders, with red and yellow areas showing the highest concentrations. Along the U.S.-Mexico border, the most intense activity is observed in California, Arizona, and Texas, reflecting significant traffic volume and economic activity. In contrast, the U.S.-Canada border exhibits high densities near the Pacific Northwest, but the crossings are more evenly distributed than the southern border. In Alaska, two hotspots are visible along the Alaska-Yukon border, with the southeastern region showing the highest density, while a smaller hotspot appears further north. Overall, the map spotlights California, Texas, the Pacific Northwest, and southeastern Alaska as the most active areas for border crossings, with minimal activity in other parts of Alaska.")

    elif selected_plot == "12. Seasonal Trend: State-wise Traffic":
        data['Date'] = pd.to_datetime(data['Date'], format="%b %Y", errors='coerce')
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
        states = data['State'].unique()
        st.subheader("Select a State to View Seasonal Trends")
        selected_state = st.selectbox("State", states)

        if selected_state:
            state_season_trend = data[data['State'] == selected_state].groupby('Season')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            state_season_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Seasonal Trend for {selected_state}", fontsize=16)
            ax.set_xlabel("Season", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    elif selected_plot == "13. Seasonal Trend: Measure-wise Traffic":
        data['Date'] = pd.to_datetime(data['Date'], format="%b %Y", errors='coerce')
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
        measures = data['Measure'].unique()
        st.subheader("Select a Measure to View Seasonal Trends")
        selected_measure = st.selectbox("Measure", measures)

        if selected_measure:
            measure_season_trend = data[data['Measure'] == selected_measure].groupby('Season')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            measure_season_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Seasonal Trend for {selected_measure}", fontsize=16)
            ax.set_xlabel("Season", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    elif selected_plot == "14. Monthly Trend: State-wise Traffic":
        data['Date'] = pd.to_datetime(data['Date'], format="%b %Y", errors='coerce')
        data['Month'] = data['Date'].dt.month
        states = data['State'].unique()
        st.subheader("Select a State to View Monthly Trends")
        selected_state = st.selectbox("State", states)
        if selected_state:
            state_monthly_trend = data[data['State'] == selected_state].groupby('Month')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            state_monthly_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Monthly Trend for {selected_state}", fontsize=16)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)


    elif selected_plot == "15. Monthly Trend: Measure-wise Traffic":
        data['Date'] = pd.to_datetime(data['Date'], format="%b %Y", errors='coerce')
        data['Month'] = data['Date'].dt.month
        measures = data['Measure'].unique()
        st.subheader("Select a Measure to View Monthly Trends")
        selected_measure = st.selectbox("Measure", measures)
        if selected_measure:
            measure_monthly_trend = data[data['Measure'] == selected_measure].groupby('Month')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            measure_monthly_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Monthly Trend for {selected_measure}", fontsize=16)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)


    elif selected_plot == "16. Yearly Trend: State-wise Traffic":
        data['Date'] = pd.to_datetime(data['Date'], format="%b %Y", errors='coerce')
        data['Year'] = data['Date'].dt.year
        states = data['State'].unique()
        st.subheader("Select a State to View Yearly Trends")
        selected_state = st.selectbox("State", states)

        if selected_state:
            state_yearly_trend = data[data['State'] == selected_state].groupby('Year')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            state_yearly_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Yearly Trend for {selected_state}", fontsize=16)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    elif selected_plot == "17. Yearly Trend: Measure-wise Traffic":
        data['Date'] = pd.to_datetime(data['Date'], format="%b %Y", errors='coerce')
        data['Year'] = data['Date'].dt.year
        measures = data['Measure'].unique()

        st.subheader("Select a Measure to View Yearly Trends")
        selected_measure = st.selectbox("Measure", measures)

        if selected_measure:
            measure_yearly_trend = data[data['Measure'] == selected_measure].groupby('Year')['Value'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            measure_yearly_trend.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Yearly Trend for {selected_measure}", fontsize=16)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Total Crossings", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    elif selected_plot == "18. Vehicular Traffic vs Pedestrian Traffic":
        pedestrian_traffic = data[data['Measure'].str.contains('Pedestrian', na=False)]['Value'].sum()
        vehicular_traffic = data[~data['Measure'].str.contains('Pedestrian', na=False)]['Value'].sum()

        traffic_data = pd.DataFrame({
            'Traffic Type': ['Vehicular Traffic', 'Pedestrian Traffic'],
            'Total Crossings': [vehicular_traffic, pedestrian_traffic]
        })

        st.subheader("Comparison of Vehicular and Pedestrian Traffic")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(traffic_data['Traffic Type'], traffic_data['Total Crossings'], color=['skyblue', 'orange'])
        ax.set_title("Vehicular Traffic vs Pedestrian Traffic", fontsize=16)
        ax.set_xlabel("Traffic Type", fontsize=12)
        ax.set_ylabel("Total Crossings", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.text("The bar chart compares the number of border crossings for vehicular and pedestrian traffic. Vehicular traffic, which includes personal vehicles, trucks, buses, and other motorized transport, accounts for the vast majority of crossings, exceeding 1.0 × 10¹⁰ total crossings over this period. In contrast, pedestrian traffic is significantly lower, with a value almost negligible compared to vehicular crossings. This indicates that border movement is predominantly vehicle-based, reflecting the heavy reliance on personal and commercial vehicles for cross-border travel and trade.")

    elif selected_plot == "19. State-wise Heatmap":
        states = data['State'].unique()
        st.subheader("Select a State to View its Border Crossings Heatmap")
        selected_state = st.selectbox("State", states)

        if selected_state:
            state_data = data[data['State'] == selected_state]
            state_map = folium.Map(location=[state_data['Latitude'].mean(), state_data['Longitude'].mean()],zoom_start=6)

            heat_data = state_data[['Latitude', 'Longitude', 'Value']].dropna().values.tolist()
            HeatMap(heat_data).add_to(state_map)

            st_folium(state_map, width=800, height=600)

    else:
        st.info("Please select a plot to display.")

with tab3:
    st.header(f"{chr(128640)} Machine Learning Models on Border Crossing Data")

    import seaborn as sns
    from sklearn.preprocessing import StandardScaler as SklearnScaler
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans as PySparkKMeans
    from sklearn.cluster import DBSCAN
    import hdbscan
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import IsolationForest
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from sklearn.preprocessing import MinMaxScaler


    spark = SparkSession.builder.appName("BorderCrossingML").getOrCreate()
    df = spark.read.csv("Border_Crossing_Entry_Data.csv", header=True, inferSchema=True)

    ml_algorithms = ["", "K-Means", "DBSCAN", "HDBSCAN", "Gaussian Mixture Model", "Isolation Forest", "Autoencoders", "LSTM"]
    selected_algo = st.selectbox("Select Machine Learning Algorithm", ml_algorithms)

    if selected_algo == "K-Means":
        df = df.dropna(subset=["Latitude", "Longitude"])

        assembler = VectorAssembler(inputCols=["Latitude", "Longitude"], outputCol="features")
        vector_df = assembler.transform(df)

        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
        scaler_model = scaler.fit(vector_df)
        scaled_df = scaler_model.transform(vector_df)

        sample_fraction = st.slider("Select Sample Size (%)", min_value=0.05, max_value=1.0, value=0.05, step=0.01)
        sample_scaled_df = scaled_df.sample(fraction=sample_fraction)

        st.subheader("📍 K-Means Clustering using PySpark")
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)
        if st.button("Run K-Means Clustering"):
            kmeans = PySparkKMeans(k=num_clusters, seed=1, featuresCol="scaledFeatures", predictionCol="cluster")
            model = kmeans.fit(sample_scaled_df)
            result = model.transform(sample_scaled_df)

            pandas_df = result.select("Latitude", "Longitude", "cluster").toPandas()

            st.session_state["kmeans_labels"] = pandas_df[["cluster"]].rename(columns={"cluster": "Kmeans_Cluster"})

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pandas_df, x="Longitude", y="Latitude", hue="cluster", palette="viridis", ax=ax)
            ax.set_title("K-Means Clustering of Border Crossings (PySpark)", fontsize=16)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            st.pyplot(fig)


            st.subheader(f"{chr(128207)} Clustering Performance Metrics")
            features = pandas_df[['Latitude', 'Longitude']].values
            labels = pandas_df['cluster']
            silhouette = silhouette_score(features, labels)
            db_index = davies_bouldin_score(features, labels)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-box'><h4>Silhouette Score</h4><p>" + str(round(silhouette, 3)) + "</p><p style='color:green'>(Higher is better)</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-box'><h4>Davies-Bouldin Index</h4><p>" + str(round(db_index, 3)) + "</p><p style='color:green'>(Lower is better)</p></div>", unsafe_allow_html=True)

    elif selected_algo == "DBSCAN":
        st.subheader(f"{chr(128205)} DBSCAN Clustering using scikit-learn")

        sample_fraction = st.slider("Select Sample Size (%)", min_value=0.05, max_value=1.0, value=0.05, step=0.01)
        sample_df = df.select("Latitude", "Longitude", "Date").dropna(subset=["Latitude", "Longitude", "Date"]).sample(fraction=sample_fraction)
        pandas_df = sample_df.toPandas()

        scaler = SklearnScaler()
        scaled_data = scaler.fit_transform(pandas_df[['Latitude', 'Longitude']])

        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.3, step=0.1)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=3)

        if st.button("Run DBSCAN Clustering"):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(scaled_data)

            pandas_df['DBSCAN_Cluster'] = labels
            pandas_df['Date'] = pd.to_datetime(pandas_df['Date'], errors='coerce')

            st.session_state["dbscan_labels"] = pandas_df[["Date", "DBSCAN_Cluster"]]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pandas_df, x="Longitude", y="Latitude", hue="DBSCAN_Cluster", palette="tab10", ax=ax)
            ax.set_title("DBSCAN Clustering of Border Crossings")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader(f"{chr(128207)} Clustering Performance Metrics")
            mask = labels != -1
            if np.sum(mask) > 1:
                sil_score = silhouette_score(scaled_data[mask], labels[mask])
                db_score = davies_bouldin_score(scaled_data[mask], labels[mask])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='metric-box'><h4>Silhouette Score</h4><p>" + str(round(sil_score, 3)) + "</p><p style='color:green'>(Higher is better)</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='metric-box'><h4>Davies-Bouldin Index</h4><p>" + str(round(db_score, 3)) + "</p><p style='color:green'>(Lower is better)</p></div>", unsafe_allow_html=True)
            else:
                st.warning("Too many noise points to calculate metrics.")


    elif selected_algo == "HDBSCAN":
        st.subheader(f"{chr(128205)} HDBSCAN Clustering using scikit-learn")

        sample_fraction = st.slider("Select Sample Size (%)", min_value=0.05, max_value=1.0, value=0.05, step=0.01)
        sample_df = df.select("Latitude", "Longitude", "Date").dropna(subset=["Latitude", "Longitude", "Date"]).sample(fraction=sample_fraction)
        pandas_df = sample_df.toPandas()


        scaler = SklearnScaler()
        scaled_data = scaler.fit_transform(pandas_df[['Latitude', 'Longitude']])

        min_cluster_size = st.slider("Minimum Cluster Size", min_value=5, max_value=50, value=5)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=3)

        if st.button("Run HDBSCAN Clustering"):
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = hdb.fit_predict(scaled_data)

            pandas_df['HDBSCAN_Cluster'] = labels
            pandas_df['Date'] = pd.to_datetime(pandas_df['Date'], errors='coerce')

            st.session_state["hdbscan_labels"] = pandas_df[["Date", "HDBSCAN_Cluster"]]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=pandas_df,
                x="Longitude",
                y="Latitude",
                hue="HDBSCAN_Cluster",
                palette="tab10",
                ax=ax,
                legend=False
            )
            ax.set_title("HDBSCAN Clustering of Border Crossings")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)

            unique_clusters = pandas_df["HDBSCAN_Cluster"].unique()
            clusters_only = [c for c in unique_clusters if c != -1]
            num_clusters = len(clusters_only)

            ax.text(
                0.95,
                0.05,
                f"Number of Clusters: {num_clusters}",
                transform=ax.transAxes,
                fontsize=12,
                color="black",
                ha="right",
                va="bottom"
            )

            st.pyplot(fig)

            st.subheader(f"{chr(128207)} Clustering Performance Metrics")
            mask = labels != -1
            if np.sum(mask) > 1:
                sil_score = silhouette_score(scaled_data[mask], labels[mask])
                db_score = davies_bouldin_score(scaled_data[mask], labels[mask])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='metric-box'><h4>Silhouette Score</h4><p>" + str(round(sil_score, 3)) + "</p><p style='color:green'>(Higher is better)</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='metric-box'><h4>Davies-Bouldin Index</h4><p>" + str(round(db_score, 3)) + "</p><p style='color:green'>(Lower is better)</p></div>", unsafe_allow_html=True)
            else:
                st.warning("Too many noise points to calculate metrics.")


    elif selected_algo == "Gaussian Mixture Model":
        st.subheader("📍 Gaussian Mixture Model Clustering")

        sample_fraction = st.slider("Select Sample Size (%)", min_value=0.05, max_value=1.0, value=0.05, step=0.01)
        sample_df = df.select("Latitude", "Longitude", "Date").dropna(subset=["Latitude", "Longitude", "Date"]).sample(fraction=sample_fraction)
        pandas_df = sample_df.toPandas()

        scaler = SklearnScaler()
        scaled_data = scaler.fit_transform(pandas_df[['Latitude', 'Longitude']])

        n_components = st.slider("Number of Components", min_value=2, max_value=10, value=3)

        if st.button("Run GMM Clustering"):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(scaled_data)
            labels = gmm.predict(scaled_data)
            probs = gmm.predict_proba(scaled_data)

            pandas_df['GMM_Cluster'] = labels
            pandas_df['Date'] = pd.to_datetime(pandas_df['Date'], errors='coerce')

            st.session_state["gmm_labels"] = pandas_df[["Date", "GMM_Cluster"]]

            pandas_df['Max_Probability'] = probs.max(axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pandas_df, x="Longitude", y="Latitude", hue="GMM_Cluster", palette="tab10", ax=ax)
            ax.set_title("Gaussian Mixture Model Clustering (Border Crossings)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader(f"{chr(128207)} Clustering Performance Metrics")
            sil_score = silhouette_score(scaled_data, labels)
            db_index = davies_bouldin_score(scaled_data, labels)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-box'><h4>Silhouette Score</h4><p>" + str(round(sil_score, 3)) + "</p><p style='color:green'>(Higher is better)</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-box'><h4>Davies-Bouldin Index</h4><p>" + str(round(db_index, 3)) + "</p><p style='color:green'>(Lower is better)</p></div>", unsafe_allow_html=True)


    elif selected_algo == "Isolation Forest":
        st.subheader(f"{chr(128680)} Isolation Forest for Anomaly Detection")

        sample_fraction = st.slider("Select Sample Size (%)", min_value=0.05, max_value=1.0, value=0.05, step=0.01)
        sample_df = df.select("Latitude", "Longitude", "Date").dropna(subset=["Latitude", "Longitude", "Date"]).sample(fraction=sample_fraction)
        pandas_df = sample_df.toPandas()


        scaler = SklearnScaler()
        scaled_data = scaler.fit_transform(pandas_df[['Latitude', 'Longitude']])

        contamination = st.slider("Contamination (Anomaly Proportion)", min_value=0.01, max_value=0.2, value=0.05)
        if st.button("Run Isolation Forest"):
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            labels = iso_forest.fit_predict(scaled_data)
            pandas_df['Anomaly'] = labels

            st.session_state["anomaly_labels"] = pandas_df[["Date", "Anomaly"]]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pandas_df, x="Longitude", y="Latitude", hue="Anomaly", palette={1: 'blue', -1: 'red'}, ax=ax)
            ax.set_title("Isolation Forest Anomaly Detection (Border Crossings)", fontsize=16)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader(f"{chr(128202)} Anomaly Detection Summary")

            anomaly_count = (labels == -1).sum()
            normal_count = (labels == 1).sum()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class='metric-box'>
                        <h4>🚨 Anomalies Detected</h4>
                        <p>{anomaly_count:,}</p>
                        <p style='color:red'>(Flagged as Outliers)</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class='metric-box'>
                        <h4>✅ Normal Points</h4>
                        <p>{normal_count:,}</p>
                        <p style='color:green'>(Inlier Observations)</p>
                    </div>
                """, unsafe_allow_html=True)



    elif selected_algo == "Autoencoders":
        st.subheader("🤖 Autoencoder for Anomaly Detection")

        sample_fraction = st.slider("Select Sample Size (%)", min_value=0.05, max_value=1.0, value=0.05, step=0.01)
        sample_df = df.select("Latitude", "Longitude", "Date").dropna(subset=["Latitude", "Longitude", "Date"]).sample(fraction=sample_fraction)
        pandas_df = sample_df.toPandas()

        scaler = SklearnScaler()
        scaled_data = scaler.fit_transform(pandas_df[['Latitude', 'Longitude']])

        input_dim = scaled_data.shape[1]

        if st.button("Run Autoencoders"):
            autoencoder = Sequential([
                Dense(32, activation="relu", input_shape=(input_dim,)),
                Dense(16, activation="relu"),
                Dense(32, activation="relu"),
                Dense(input_dim, activation="linear")
            ])

            autoencoder.compile(optimizer="adam", loss="mse")
            autoencoder.fit(scaled_data, scaled_data, epochs=20, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)

            reconstructed = autoencoder.predict(scaled_data)
            reconstruction_error = np.mean((scaled_data - reconstructed) ** 2, axis=1)
            threshold = st.slider("Anomaly Threshold", min_value=0.01, max_value=1.0, value=0.1)
            anomalies = (reconstruction_error > threshold).astype(int)
            pandas_df['Anomaly'] = anomalies
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pandas_df, x="Longitude", y="Latitude", hue="Anomaly", palette={0: 'blue', 1: 'red'},ax=ax)

            ax.set_title("Autoencoder Anomaly Detection (Border Crossings)", fontsize=16)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("📊 Anomaly Detection Summary")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class='metric-box'>
                        <h4>🚨 Anomalies Detected</h4>
                        <p>{(anomalies == 1).sum():,}</p>
                        <p style='color:red'>(Flagged as Outliers)</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class='metric-box'>
                        <h4>✅ Normal Points</h4>
                        <p>{(anomalies == 0).sum():,}</p>
                        <p style='color:green'>(Inlier Observations)</p>
                    </div>
                """, unsafe_allow_html=True)
    elif selected_algo == "LSTM":
        st.subheader("\U0001F4C8 Simple LSTM Forecasting of Border Crossings")

        df = df.dropna(subset=["Value"])

        pandas_df = df.select("Date", "Value").toPandas()

        pandas_df["Date"] = pd.to_datetime(pandas_df["Date"], format="%b %Y", errors='coerce')

        pandas_df = pandas_df.dropna().sort_values("Date")

        grouped_df = pandas_df.groupby("Date")["Value"].sum().reset_index()

        scaler = MinMaxScaler()

        grouped_df[["Value"]] = scaler.fit_transform(grouped_df[["Value"]])


        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:i + seq_length]
                y = data[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)


        seq_length = st.slider("Sequence Length", min_value=5, max_value=30, value=12)
        data_seq = grouped_df[["Value"]].values

        col1, col2 = st.columns(2)
        with col1:
            run_simple = st.button("LSTM Model")

        if run_simple:
            X, y = create_sequences(data_seq, seq_length)
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            predictions = model.predict(X)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y.reshape(-1, 1))
            result_df = grouped_df.iloc[seq_length:].copy()
            result_df["Predicted"] = predictions
            result_df["Actual"] = actual

            st.line_chart(result_df.set_index("Date")[['Actual', 'Predicted']])

            from sklearn.metrics import mean_squared_error, mean_absolute_error

            mse = mean_squared_error(actual, predictions)


            mae = mean_absolute_error(actual, predictions)

            st.subheader("\U0001F4CF Forecasting Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                    <div class='metric-box'>
                        <h4>\U0001F4C9 Mean Squared Error (MSE)</h4>
                        <p>{mse:.3f}</p>
                        <p style='color:red'>(Lower is better)</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class='metric-box'>
                        <h4>\U0001F4C9 Mean Absolute Error (MAE)</h4>
                        <p>{mae:.3f}</p>
                        <p style='color:red'>(Lower is better)</p>
                    </div>
                """, unsafe_allow_html=True)


