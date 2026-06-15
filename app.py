import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍 Customer Segmentation with Clustering")
# ==========================================================
# LOAD ARTIFACTS
# ==========================================================

@st.cache_data
def load_data():   
    df = pd.read_excel("Online_Retail.xlsx")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    rfm = pd.read_csv("rfm_dataset.csv")
    comparison = pd.read_csv("algorithm_comparison.csv")
    profiles = pd.read_csv("cluster_profiles.csv")
    profiles.rename(columns={"Unnamed: 0": "Cluster"}, inplace=True)
    pca_df = pd.read_csv("pca_data.csv")

    return df, rfm, comparison, profiles, pca_df


@st.cache_resource
def load_models():
    artifacts = joblib.load("artifacts.pkl")

    return artifacts

data = load_data()
df, rfm, comparison, profiles, pca_df= load_data()
artifacts = load_models()
kmeans=artifacts["kmeans"]
scaler=artifacts["scaler"]
pca=artifacts["pca"]
best_k=artifacts["best_k"]
# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🛍️ Customer Segmentation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Overview",
        "RFM Analysis",
        "Elbow & Silhouette",
        "Algorithm Comparison",
        "Cluster Visualization",
        "Cluster Profiles",
        "Segment Predictor",
        "Data Analysis",
        "Business Recommendations"
    ]
)

cluster_choice = st.sidebar.selectbox("Highlight Cluster",["All"] + list(sorted(rfm["Cluster"].unique())))
plot_df = pca_df.copy()

if cluster_choice != "All":
    plot_df = plot_df[
        plot_df["Cluster"] == cluster_choice
    ]

recency_range = st.sidebar.slider(
    "Recency Range",
    int(rfm["Recency"].min()),
    int(rfm["Recency"].max()),
    (
        int(rfm["Recency"].min()),
        int(rfm["Recency"].max())
    )
)

frequency_range = st.sidebar.slider(
    "Frequency Range",
    int(rfm["Frequency"].min()),
    int(rfm["Frequency"].max()),
    (
        int(rfm["Frequency"].min()),
        int(rfm["Frequency"].max())
    )
)

monetary_range = st.sidebar.slider(
    "Monetary Range",
    float(rfm["Monetary"].min()),
    float(rfm["Monetary"].max()),
    (
        float(rfm["Monetary"].min()),
        float(rfm["Monetary"].max())
    )
)

filtered_rfm = rfm[
(rfm["Recency"].between(*recency_range)) &
    (rfm["Frequency"].between(*frequency_range)) &
    (rfm["Monetary"].between(*monetary_range))     
]
dynamic_profiles = (filtered_rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean().round(2).reset_index())

col1, col2, col3=st.columns(3)
with col1:
     st.metric(
     "Customers Selected",
     len(filtered_rfm)
     )
with col2:
     st.metric(
     "Average Monetary",
     f"₹{filtered_rfm['Monetary'].mean():,.0f}"
     )
with col3:
     st.metric(
     "Average Frequency",
     f"{filtered_rfm['Frequency'].mean():.1f}"
     )

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

if page == "Project Overview":

    st.title("Project Overview")

    st.markdown("""
    ### Project Objective

    Segment customers based on purchasing behaviour using
    **RFM Analysis** and **Clustering Techniques**.

    This project evaluates multiple clustering algorithms:

    - ✅ K-Means
    - ✅ Hierarchical Clustering
    - ✅ DBSCAN

    and selects the optimal model for deployment.

    ### Methodology

    1. Data Cleaning
    2. RFM Feature Engineering
    3. Log Transformation
    4. Standardization
    5. Optimal K Selection
    6. Clustering
    7. Model Evaluation
    8. Business Recommendations
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
         st.metric(
         "Customers",
         len(filtered_rfm)
         )
    with col2:
         st.metric(
         "Optimal K",
         best_k 
         )
    with col3:
         st.metric(
         "Algorithms Compared",
         3
         )

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
         st.metric(
         "Total Customers",
         len(filtered_rfm)
         ) 
    with col2:
         st.metric(
         "Features",
         4
         )
    with col3:
         st.metric(
         "Clusters",
         len(filtered_rfm["Cluster"].unique())
         )

    st.subheader("RFM Dataset")

    st.dataframe(
        filtered_rfm.head(20),
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        filtered_rfm.describe(),
        use_container_width=True
    )

# ==========================================================
# RFM ANALYSIS
# ==========================================================

elif page == "RFM Analysis":

    st.title("📈 RFM Analysis")

    tab1, tab2, tab3 = st.tabs(
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    )

    with tab1:

        st.subheader("Recency Distribution")

        fig = px.histogram(
            filtered_rfm,
            x="Recency",
            nbins=30,
            title="Recency Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Lower Recency indicates more recent purchases."
        )

    with tab2:

        st.subheader("Frequency Distribution")

        fig = px.histogram(
            filtered_rfm,
            x="Frequency",
            nbins=30,
            title="Frequency Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Higher Frequency indicates repeat customers."
        )

    with tab3:

        st.subheader("Monetary Distribution")

        fig = px.histogram(
            filtered_rfm,
            x="Monetary",
            nbins=30,
            title="Monetary Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Higher Monetary value indicates high-spending customers."
        )

# ==========================================================
# ELBOW & SILHOUETTE ANALYSIS
# ==========================================================

elif page == "Elbow & Silhouette":

    st.title("📉 Cluster Selection Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Elbow Method")

        try:
            elbow_img = Image.open("elbow_plot.png")
            st.image(
                elbow_img,
                use_container_width=True
            )
        except:
            st.warning("elbow_plot.png not found.")

        st.markdown("""
        **Interpretation:**

        - Shows how inertia decreases as K increases.
        - The 'elbow' indicates diminishing returns.
        - Helps identify an appropriate number of clusters.
        """)

    with col2:

        st.subheader("Silhouette Analysis")

        try:
            sil_img = Image.open("silhouette_plot.png")
            st.image(
                sil_img,
                use_container_width=True
            )
        except:
            st.warning("silhouette_plot.png not found.")

        st.markdown(f"""
        **Optimal K selected during training:** `{best_k}`

        The silhouette score measures cluster separation.

        Higher values indicate better-defined clusters.
        """)

# ==========================================================
# ALGORITHM COMPARISON
# ==========================================================

elif page == "Algorithm Comparison":

    st.title("⚖️ Algorithm Comparison")

    st.subheader("Evaluation Metrics")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            comparison,
            x="Algorithm",
            y="Silhouette Score",
            title="Silhouette Score Comparison",
            text_auto=".3f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            comparison,
            x="Algorithm",
            y="Davies-Bouldin Index",
            title="Davies-Bouldin Index Comparison",
            text_auto=".3f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info("""
    Interpretation:

    • Higher Silhouette Score is better.

    • Lower Davies-Bouldin Index is better.

    • K-Means was selected for deployment because
      it supports prediction for new customers.
    """)

# ==========================================================
# PCA CLUSTER VISUALIZATION
# ==========================================================

elif page == "Cluster Visualization":

    st.title("🎯 PCA Cluster Visualization")

    st.markdown("""
    PCA reduces the three-dimensional RFM space into
    two dimensions for visualization.
    """)

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color=plot_df["Cluster"].astype(str),
        title="Customer Clusters (PCA Projection)",
        labels={"color": "Cluster"},
        hover_data=["Cluster"]
    )

    fig.update_traces(
        marker=dict(size=7)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        f"The final deployed K-Means model identified {best_k} customer clusters."
    )

    st.subheader("Cluster Distribution")

    cluster_counts = (
        filtered_rfm["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Customers"
    ]

    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Customers",
        title="Customer Distribution Across Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# CLUSTER PROFILES
# ==========================================================

elif page == "Cluster Profiles":

    st.title("👥 Cluster Profiles")

    st.subheader("Average RFM Values by Cluster")

    st.dataframe(
        dynamic_profiles,
        use_container_width=True
    )

    st.markdown("### Segment Insights")

    for _, row in dynamic_profiles.iterrows():

        cluster = row["Cluster"]
        with st.expander(f"Cluster {cluster}"):

            st.write(f"**Recency:** {row['Recency']:.2f}")
            st.write(f"**Frequency:** {row['Frequency']:.2f}")
            st.write(f"**Monetary:** ₹{row['Monetary']:.2f}")

# ==========================================================
# CUSTOMER SEGMENT PREDICTOR
# ==========================================================

elif page == "Segment Predictor":

    st.title("🔮 Customer Segment Predictor")

    st.markdown("""
    Enter customer RFM values to predict the customer segment.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30
        )

    with col2:
        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5
        )

    with col3:
        monetary = st.number_input(
            "Monetary Value",
            min_value=0.0,
            value=1000.0
        )

    if st.button("Predict Segment"):

        input_df = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        # Log Transform
        input_log = np.log1p(input_df)

        # Scaling
        input_scaled = scaler.transform(input_log)

        # Prediction
        cluster = kmeans.predict(input_scaled)[0]

        st.success(f"Predicted Cluster: {cluster}")

        try:
            segment_name = profiles.loc[
                profiles["Cluster"] == cluster,
                "Segment"
            ].values[0]

            st.info(f"Segment: **{segment_name}**")

        except:
            st.warning("Segment description unavailable.")

# ==========================================================
# BUSINESS RECOMMENDATIONS
# ==========================================================
elif page == "Data Analysis":
    monthly_revenue = (
    df.groupby("YearMonth")["Revenue"]
    .sum()
    .reset_index()
    )

    fig = px.line(
    monthly_revenue,
    x="YearMonth",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
   
    monthly_orders = (
    df.groupby("YearMonth")["InvoiceNo"]
    .nunique()
    .reset_index(name="Orders")
    )

    fig = px.line(
    monthly_orders,
    x="YearMonth",
    y="Orders",
    markers=True,
    title="Monthly Orders Trend"
    )
 
    st.plotly_chart(fig, use_container_width=True)

    monthly_customers = (
    df.groupby("YearMonth")["CustomerID"]
    .nunique()
    .reset_index(name="Active Customers")
    )

    fig = px.line(
    monthly_customers,
    x="YearMonth",
    y="Active Customers",
    markers=True,
    title="Monthly Active Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

    country_sales = (
    df.groupby("Country")["Revenue"]
    .sum()
    .nlargest(10)
    .reset_index()
    )

    fig = px.bar(
    country_sales,
    x="Revenue",
    y="Country",
    orientation="h",
    title="Top 10 Countries by Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    segment_revenue = (
    filtered_rfm.groupby("Cluster")["Monetary"]
    .sum()
    .reset_index()
    )

    fig = px.bar(
    segment_revenue,
    x="Cluster",
    y="Monetary",
    title="Revenue Contribution by Segment"
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "Business Recommendations":

    st.title("💡 Business Recommendations")

    for _, row in profiles.iterrows():

        cluster = row["Cluster"]
        segment = row["Segment"]

        st.subheader(f"Cluster {cluster}: {segment}")

        segment_lower = str(segment).lower()

        if "vip" in segment_lower:

            st.markdown("""
            - Offer premium loyalty programs.
            - Provide exclusive discounts.
            - Early access to products.
            - Personalized experiences.
            """)

        elif "loyal" in segment_lower:

            st.markdown("""
            - Encourage referrals.
            - Upsell complementary products.
            - Maintain engagement.
            - Reward repeat purchases.
            """)

        elif "risk" in segment_lower:

            st.markdown("""
            - Launch win-back campaigns.
            - Offer time-limited promotions.
            - Collect feedback.
            - Re-engage through email marketing.
            """)

        elif "potential" in segment_lower:

            st.markdown("""
            - Promote personalized offers.
            - Increase interaction frequency.
            - Recommend relevant products.
            - Encourage repeat purchases.
            """)

        else:

            st.markdown("""
            - Monitor customer behavior.
            - Explore targeted campaigns.
            - Improve engagement.
            - Identify growth opportunities.
            """)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    **Customer Segmentation Dashboard**

    Built using:
    - Streamlit
    - Scikit-learn
    - Plotly
    - RFM Analysis
    - K-Means, Hierarchical Clustering, and DBSCAN

    This project demonstrates end-to-end unsupervised machine learning,
    model comparison, customer analytics, and deployment readiness.
    """
)
