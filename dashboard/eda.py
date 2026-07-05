import streamlit as st
import pandas as pd
from utils import load_raw_dataset, load_clustered_dataset 
import plotly.express as px
import plotly.graph_objects as go

def eda_page():
    st.title("📊 Exploratory Data Analysis")

    # Load dataset
    
    df = load_clustered_dataset()

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    
    st.sidebar.header("EDA Filters")

    feature = st.sidebar.selectbox(
"Feature",["Recency", "Frequency", "Monetary","CustomerValue","AverageOrderValue"],key="eda_feature_filter")

    selected_clusters =st.sidebar.multiselect ("Select Cluster", sorted (df["Cluster"].unique()),default=sorted(df["Cluster"].unique()),key="cluster_filter")

    selected_features =st.sidebar.multiselect ("Distribution Features",["Recency", "Frequency","Monetary"],default=["Recency","Frequency","Monetary"],key="distribution_feature")

    min_monetary = st.sidebar.slider(
    "Minimum Monetary",
    float(df["Monetary"].min()),
    float(df["Monetary"].max()),
    float(df["Monetary"].min())
    )

    min_frequency = st.sidebar.slider(
    "Minimum Frequency",
    int(df["Frequency"].min()),
    int(df["Frequency"].max()),
    int(df["Frequency"].min())
    )

    max_recency = st.sidebar.slider(
    "Maximum Recency",
    float(df["Recency"].min()),
    float(df["Recency"].max()),
    float(df["Recency"].max())
    )

    if "AverageOrderValue" not in df.columns:
       df["AverageOrderValue"] = (df["Monetary"] / df["Frequency"].replace(0, 1))

    if "CustomerValue" not in df.columns:
       df["CustomerValue"] = (df["Frequency"] * df["AverageOrderValue"])

    top_n = st.sidebar.slider("Top Customers",10,
100,20)

    filtered_df = df[
    (df["Cluster"].isin(selected_clusters)) &
    (df["Monetary"] >= min_monetary) &
    (df["Frequency"] >= min_frequency) &
    (df["Recency"] <= max_recency)
    ].copy()
   
    # -----------------------------
    # KPI Cards
    # -----------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Customers",len(filtered_df))

    col2.metric("Avg Monetary",f"{filtered_df['Monetary'].mean():.2f}")

    col3.metric("Avg Frequency",f"{filtered_df['Frequency'].mean():.2f}")

    col4.metric("Avg Recency",f"{filtered_df['Recency'].mean():.2f}")

    col5.metric("Avg ",f"{filtered_df['CustomerValue'].mean():.2f}")

    st.divider()

    # ==========================================
    # Cluster Distribution
    # ==========================================

    st.subheader("Customer Distribution by Cluster")

    cluster_counts = (filtered_df["Cluster"].value_counts().sort_index().reset_index())

    cluster_counts.columns = ["Cluster", "Customers"]

    fig = px.bar(cluster_counts,x="Cluster",y="Customers",color="Cluster",text="Customers",title="Customers in Each Cluster")

    fig.update_layout(xaxis_title="Cluster",yaxis_title="Number of Customers",template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"📊 {feature} Distribution")

    fig = px.histogram(filtered_df,x=feature, color="Cluster",marginal="box",nbins=35)

    fig.update_layout(template="plotly_dark",
height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =====================================================
    # RFM SCATTER PLOTS
    # =====================================================

    st.header("📈 RFM Relationship Analysis")

    plot_df = filtered_df.copy()

    plot_df["Monetary_Size"] = plot_df["Monetary"].abs() + 1
    plot_df["Recency_Size"] = plot_df["Recency"].abs() + 1
    plot_df["Frequency_Size"] = plot_df["Frequency"].abs() + 1

    col1, col2 = st.columns(2)

    with col1:

         fig = px.scatter(plot_df,x= "Recency_Size",y="Frequency",color="Cluster",
size="Monetary_Size",hover_data=["Recency", "Frequency_Size", "Monetary"],title="Recency vs Frequency")

         fig.update_layout (template="plotly_dark")

         st.plotly_chart(fig, use_container_width=True)

    with col2:

         fig = px.scatter(plot_df,x= "Frequency_Size",y="Monetary_Size",color="Cluster",size= "Recency_Size", hover_data= ["Recency","Frequency", "Monetary"], title="Frequency vs Monetary")

         fig.update_layout (template="plotly_dark")

         st.plotly_chart(fig, use_container_width=True)

    # =====================================
    # BUBBLE CHART
    # =====================================

    st.markdown("---")    

    st.subheader("Customer Bubble Analysis")

    x_axis = st.selectbox("X Axis", ["Recency","Frequency","Monetary"],key="bubble_x")

    y_axis = st.selectbox("Y Axis",
["Recency","Frequency","Monetary"],index=2,
key="bubble_y")

    size = st.selectbox("Bubble Size",
["Frequency","Monetary", "Recency"],key="bubble_size")

    plot_df = filtered_df.copy()

    plot_df["Frequency_Size"] = plot_df["Frequency"].abs() + 1

    fig = px.scatter(
    filtered_df,
    x="Recency_Size",
    y="Monetary_Size",
    size="Frequency_Size",
    color="Cluster",
    hover_data=["Recency", "Frequency", "Monetary"],
    title="Customer Segments"
    )

    fig.update_layout(
    template="plotly_dark",
    height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    corr = filtered_df[["Recency","Frequency","Monetary"]].corr()

    fig = px.imshow(corr,text_auto=True,color_continuous_scale="Blues",aspect="auto")

    fig.update_layout(template="plotly_dark",height=500)

    st.plotly_chart(fig,use_container_width=True)

    # =====================================================
    # CLUSTER SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader("Cluster Statistics")

    summary = (filtered_df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean().round(2))

    st.dataframe(summary,use_container_width=True)

    # =====================================================
    # TOP CUSTOMERS
    # =====================================================

    st.markdown("---")

    st.subheader("Top 20 High Value Customers")

    top_customers = (filtered_df.sort_values(by=feature,ascending=False).head(20))

    st.dataframe(top_customers,use_container_width=True)

    # =====================================================
    # MONETARY CONTRIBUTION
    # =====================================================

    st.markdown("---")

    st.subheader("Revenue Contribution")

    cluster_sales = (filtered_df.groupby("Cluster")[feature].sum().reset_index())

    fig = px.pie(cluster_sales,names="Cluster",values=feature,hole=.55,title="Revenue Contribution by Cluster")

    fig.update_layout(template="plotly_dark",height=600)

    st.plotly_chart(fig,use_container_width=True)

    # ==========================================================
    # ADVANCED ANALYTICS
    # ==========================================================

    st.header("📈 Advanced Business Analytics")

    # ----------------------------------------------------------
    # Treemap
    # ----------------------------------------------------------

    st.subheader("Customer Value Treemap")

    fig = px.treemap(filtered_df,path=["Cluster"],values=feature,color=feature,color_continuous_scale="Blues")

    fig.update_layout (template= "plotly_dark",height=650)

    st.plotly_chart (fig, use_container_width=True)

    # ----------------------------------------------------------
    # Box Plot
    # ----------------------------------------------------------

    st.markdown("---")

    fig= px.box(filtered_df,x="Cluster",y= feature,color="Cluster",title= f"{feature} Distribution")
    fig.update_layout (template="plotly_dark")
    st.plotly_chart (fig,use_container_width=True)

    # ----------------------------------------------------------
    # Violin Plot
    # ----------------------------------------------------------

    st.markdown("---")

    fig= px.violin (filtered_df, x="Cluster", y=feature,color="Cluster",box=True,title="Customer f"{feature}")

    fig.update_layout (template= "plotly_dark",height=600)

    st.plotly_chart (fig, use_container_width=True)

    # ----------------------------------------------------------
    # Radar Chart
    # ----------------------------------------------------------

    st.markdown("---")

    st.subheader("Cluster Profile Radar")

    radar = (filtered_df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean())

    cluster_choice = st.selectbox("Select Cluster",radar.index)

    values = radar.loc[cluster_choice]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(r=values.values,theta=["Recency","Frequency","Monetary"],fill="toself",name=f"Cluster {cluster_choice}"))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),template="plotly_dark",height=600)

    st.plotly_chart(fig,use_container_width=True)

    # ----------------------------------------------------------
    # Parallel Coordinates
    # ----------------------------------------------------------

    st.markdown("---")

    st.subheader("Parallel Coordinates")

    fig = px.parallel_coordinates(filtered_df,color="Cluster",dimensions=["Recency","Frequency","Monetary"],color_continuous_scale=px.colors.sequential.Blues)

    st.plotly_chart(fig,use_container_width=True)

    # ----------------------------------------------------------
    # Cluster Table
    # ----------------------------------------------------------

    st.markdown("---")

    st.subheader("Cluster Summary")

    summary = (filtered_df.groupby("Cluster").agg({"Recency":"mean","Frequency":"mean","Monetary":"mean"}).round(2))

    st.dataframe(summary,use_container_width=True)

    # ----------------------------------------------------------
    # Download Analytics
    # ----------------------------------------------------------

    st.markdown("---")

    csv = summary.to_csv(index=True).encode()

    st.download_button("⬇ Download Cluster Summary",csv,"cluster_summary.csv","text/csv")
