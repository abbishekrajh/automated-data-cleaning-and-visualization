import plotly.express as px

def plot_distribution(df, column):
    """Interactive histogram with a marginal box plot and hover details"""
    fig = px.histogram(
        df,
        x=column,
        marginal = "box",
        title = f"Distribution of {column}",
        template = "plotly_white",
        color_discrete_sequence = ["#1f77b4"],
    )
    fig.update_layout(bargap=0.1)
    return fig

def plot_correlation_heatmap(df):
    """Interactive correlation matrix heatmap with rounded coefficients"""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        return None 
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Numeric Correlation Matrix",
        template="plotly_white"
    )
    return fig

def plot_categorical_counts(df, column, top_n=10):
    """Interactive bar chart for top N category occurrences."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    
    fig = px.bar(
        counts,
        x=column,
        y="Count",
        text="Count",
        title=f"Top {top_n} Categories in {column}",
        template="plotly_white",
        color_discrete_sequence=["#2ca02c"]
    )
    fig.update_traces(textposition="outside")
    return fig