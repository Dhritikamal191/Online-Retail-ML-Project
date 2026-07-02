import streamlit as st

def page_title(title, subtitle=""):

    st.title(title)

    if subtitle:

        st.caption(subtitle)


def metric_card(title, value):

    st.metric(

        title,

        value

    )


def section(title):

    st.markdown("---")

    st.subheader(title)


def success(msg):

    st.success(msg)


def warning(msg):

    st.warning(msg)


def info(msg):

    st.info(msg)


def error(msg):

    st.error(msg)