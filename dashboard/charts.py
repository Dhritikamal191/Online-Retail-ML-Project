import plotly.express as px
import plotly.graph_objects as go


def cluster_distribution(df):

    return px.bar(

        df.groupby("Cluster")

        .size()

        .reset_index(name="Customers"),

        x="Cluster",

        y="Customers",

        color="Cluster",

        title="Cluster Distribution"

    )


def monetary_box(df):

    return px.box(

        df,

        x="Cluster",

        y="Monetary",

        color="Cluster"

    )


def frequency_histogram(df):

    return px.histogram(

        df,

        x="Frequency",

        color="Cluster"

    )


def monetary_histogram(df):

    return px.histogram(

        df,

        x="Monetary",

        color="Cluster"

    )


def recency_histogram(df):

    return px.histogram(

        df,

        x="Recency",

        color="Cluster"

    )


def health_gauge(score):

    fig=go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=score,

            gauge={

                "axis":{"range":[0,100]}

            }

        )

    )

    return fig