import streamlit as st
import pandas as pd
from predict import predict_cluster
import plotly.graph_objects as go
from predict import predict_cluster
import os
from datetime import datetime
import requests
import plotly.express as px
from utils import show_table

def prediction_page():
    
    st.title("👥 Customer Segmentation Prediction")

    st.markdown("""
    Predict customer segments using the trained
    Machine Learning model.
    """)

    st.divider()
      

    # ==========================================
    # CUSTOMER INPUT
    # ==========================================

    st.subheader("Customer Information")

    left, right = st.columns(2)

    with left:
         recency = st.slider("Recency (Days)",0,365,30)
         frequency = st.slider("Purchase Frequency",1,100,5)
         average_order_value = st.number_input(
         "Average Order Value",
         min_value=0.0,
         value=100.0,
         step=10.0
         )

    with right:
         monetary = st.number_input("Monetary Value",min_value=1.0,value=500.0,step=50.0)
         customer_value = st.number_input(
         "Customer Value",
         min_value=0.0,
         value=500.0,
         step=50.0
         )
    st.divider()

    # ==========================================
    # INPUT SUMMARY
    # ==========================================

    st.subheader("Customer Summary")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Recency",recency)

    c2.metric("Frequency",frequency)

    c3.metric("Monetary",f"${monetary:,.2f}")

    c4.metric("CustomerValue",f"${customer_value:,.2f}")

    c5.metric("AverageOrderValue",f"${average_order_value:,.2f}")

    st.divider()

    # ==========================================
    # PREDICTION
    # ==========================================

    if st.button("🚀 Predict Customer Segment", use_container_width=True):
       
       customer = {"Recency": recency, "Frequency": frequency,"Monetary": monetary,"CustomerValue":customer_value,"AverageOrderValue":average_order_value}

       result = predict_cluster(customer)

       cluster = result["cluster"]

       segment = result["segment"]

       distance = result["distance"]

       st.success("Prediction Completed Successfully")

       col1, col2, col3 = st.columns(3)

       with col1:
            st.metric("Cluster",cluster)

       with col2:
            st.metric("Customer Segment",segment)

       with col3:
            if distance is None:
               st.metric("Distance","N/A")
            else:
                 st.metric ("Distance",round(distance,3))

       # ====================================
       # SEGMENT CARD
       # ====================================

       st.markdown("---")

       colors = {"VIP Customers":"🟢","New Customers":"🔵","At Risk Customers":"🟠","Inactive Customers":"🔴"}

       icon = colors.get(segment,"⚪")

       st.info(f"""

       # {icon} {segment}

       ### Customer Profile

       **Recency:** {recency}

       **Frequency:** {frequency}

       **Monetary:** ${monetary:,.2f}
       **CustomerValue:** ${customer_value:,.2f}
       **AverageOrderValue:** ${average_order_value:,.2f}
       """)

       # ===================================
       # CONFIDENCE GAUGE
       # ===================================

       if distance is not None:
          confidence = max(0,min (100,100- (distance*100)))

          fig = go.Figure(go.Indicator(mode="gauge+number",value=confidence,title={"text":"Prediction Confidence"},gauge={"axis":{"range":[0,100]},"bar":{"color":"royalblue"},"steps":[{"range":[0,40],"color":"red"},{"range":[40,75],"color":"orange"},{"range":[75,100],"color":"green"}]}))

          fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          
          st.plotly_chart(fig, use_container_width=True)

       # ==================================
       # CUSTOMER PERSONA
       # ==================================

       st.markdown("---")

       st.subheader("🧑 Customer Persona")

       persona = {"VIP Customers":{"Type":"High Value Customer","Description":"Frequently purchases and spends significantly.","Strategy":"Reward loyalty with premium benefits."},"New Customers":{"Type":"Recently Acquired","Description":"Still exploring products and services.","Strategy":"Provide onboarding offers and recommendations."},"At Risk Customers":{"Type":"Potential Churn","Description":"Purchase frequency is declining.","Strategy":"Launch retention campaigns and targeted discounts."},"Inactive Customers":{"Type":"Dormant Customer","Description":"No recent purchasing activity.","Strategy":"Reactivate with exclusive promotions."}}

       info = persona.get(segment)

       if info:
          st.success(f"""
          ### {segment}

          **Customer Type:** {info['Type']}

          **Description:** {info['Description']}

          **Recommended Strategy:** {info['Strategy']}
       """)

       # ===================================
       # BUSINESS RECOMMENDATIONS
       # ===================================

       st.markdown("---")

       st.subheader("💡 Recommended Business Actions")

       recommendations={"VIP Customers":["Offer Premium Membership","Exclusive Early Access","Loyalty Rewards","Cross-sell Premium Products"],"New Customers":["Welcome Discount","Email Campaign","Product Recommendations","Referral Program"],"At Risk Customers":["Retention Email","Personalized Coupon","Customer Satisfaction Survey","Special Discount"],"Inactive Customers":["Win-back Campaign","High Value Discount","Reactivate Account","Exclusive Promotion"]}

       for item in recommendations.get(segment,[]):

           st.markdown(f"✅ {item}")

       # ====================================
       # RFM RADAR CHART
       # ====================================

       st.markdown("---")

       st.subheader("📈 Customer RFM Profile")

       fig = go.Figure()

       fig.add_trace (go.Scatterpolar(r=[recency,frequency,monetary,customer_value,average_order_value/100],theta= ["Recency","Frequency","Monetary","CustomerValue","AverageOrderValue"],fill="toself",name=segment))

       fig.update_layout (polar=dict(radialaxis=dict(visible=True)),font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")

       st.plotly_chart(fig, use_container_width=True)

       # ==================================
       # SAVE PREDICTION LOG
       # ==================================

       st.markdown("---")

       log_dir="logs"

       os.makedirs(log_dir,exist_ok=True)

       log_file= os.path.join(log_dir,"prediction_logs.csv")

       prediction= pd.DataFrame({"Timestamp":[datetime.now()],"Recency":[recency],"Frequency":[frequency],"Monetary":[monetary],"Cluster":[cluster],"Segment":[segment],"Distance":[distance],"CustomerValue":[customer_value],"AverageOrderValue":[average_order_value]})

       if os.path.exists(log_file):
          prediction.to_csv (log_file,mode="a",header=False,index=False)

       else:
            prediction.to_csv (log_file,index=False)

       st.success("Prediction saved successfully.")

       # ================================
       # CURRENT PREDICTION
       # ================================

       st.markdown("---")

       st.subheader("📋 Current Prediction")

       show_table (prediction)

       # ===================================
       # PREDICTION HISTORY
       # ===================================

       st.markdown("---")

       st.header("📜 Prediction History")

       log_file = "logs/prediction_logs.csv"

       if os.path.exists(log_file):
          history = pd.read_csv(log_file)

          show_table (history.tail(20))

       else:
            st.info("No prediction history available.")

       # ==================================
       # PREDICTION STATISTICS
       # ==================================

       if os.path.exists(log_file):
          st.markdown("---")

          st.subheader("Prediction Statistics")

          col1,col2=st.columns(2)

          with col1:
               fig = px.histogram(history,x="Segment",color="Segment",title="Predicted Customer Segments")

               fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
              
               st.plotly_chart (fig,use_container_width=True)

          with col2:
               fig = px.scatter(history,x="Frequency",y="Monetary",color="Segment",size="Monetary",hover_data=["Recency"],title="Prediction Distribution")

               fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
               
               st.plotly_chart (fig,use_container_width=True)

       # ==============================
       # DOWNLOAD REPORT
       # ==============================

       st.markdown("---")

       st.subheader("Download Prediction")

       csv = prediction.to_csv(index=False).encode()

       st.download_button("⬇ Download Prediction CSV",csv,"prediction.csv","text/csv")

       # ==========================================================
       # API STATUS
       # ==========================================================

       st.markdown("---")

       st.subheader("🌐 FastAPI Status")

       try:
           response = requests.get("https://online-retail-ml-project.onrender.com/docs#",timeout=3)

           if response.status_code == 200:
              st.success("🟢 FastAPI is Running")

           else:
                st.warning("🟠 API Responded with Error")

       except:
              st.error("🔴 FastAPI Not Running")

       # ==========================================================
       # MODEL INFORMATION
       # ==========================================================

       st.markdown("---")

       st.subheader("Model Information")

       left,right=st.columns(2)

       with left:
            st.success("""
            Algorithm KMeans Clustering
            """)

       with right:
            st.success("""

            Features

            Recency

            Frequency

            Monetary

            """)

       # ==========================================================
       # SYSTEM STATUS
       # ==========================================================

       st.markdown("---")

       st.subheader("Pipeline Status")

       c1,c2,c3,c4=st.columns(4)

       c1.metric("Prediction","Ready")
       c2.metric("Model","Loaded")
       c3.metric("Logging","Enabled")
       c4.metric("Monitoring","Enabled")

       # ==========================================================
       # FOOTER
       # ==========================================================

       st.markdown("---")

       st.markdown(
       """
       <div> 
style='text-align:center;color:gray;'>

         ### Intelligent Retail Customer Segmentation Platform

         Built with ❤️ using

         Streamlit • Scikit-learn • MLflow • FastAPI • Docker • Evidently AI • GitHub Actions

       </div>
       """,unsafe_allow_html=True)
