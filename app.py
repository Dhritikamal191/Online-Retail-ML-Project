import streamlit as st

from dashboard.home import home_page
from dashboard.eda import eda_page
from dashboard.prediction import prediction_page
from dashboard.model_dashboard import model_dashboard_page
from dashboard.reports import reports_page
from dashboard.mlops import mlops_page
from dashboard.settings import settings_page

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Retail ML Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD CSS
# =====================================

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🛍️ Retail ML Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Exploratory Data Analysis",
        "🎯 Prediction",
        "🤖 Model Dashboard",
        "📄 Reports",
        "⚙️ MLOps",
        "🔧 Settings"
    ],
)

# =====================================
# PAGE ROUTING
# =====================================

if page == "🏠 Home":
    home_page()

elif page == "📊 Exploratory Data Analysis":
    eda_page()

elif page == "🎯 Prediction":
    prediction_page()

elif page == "🤖 Model Dashboard":
    model_dashboard_page()

elif page == "📄 Reports":
    reports_page()

elif page == "⚙️ MLOps":
    mlops_page()

elif page == "🔧 Settings":
    settings_page()