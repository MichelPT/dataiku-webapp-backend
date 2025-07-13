import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import gaussian_kde


def plot_histogram(df: pd.DataFrame, log_column: str, n_bins: int):
    if log_column not in df.columns:
        raise ValueError(f"Kolom '{log_column}' tidak ditemukan.")

    data = df[log_column].dropna()
    if data.empty:
        raise ValueError(f"Tidak ada data valid di kolom '{log_column}'.")

    hist_y, hist_x = np.histogram(data, bins=n_bins, density=True)
    kde = gaussian_kde(data)
    kde_x = np.linspace(hist_x.min(), hist_x.max(), 500)
    kde_y = kde(kde_x)

    cdf_x = np.sort(data)
    cdf_y = np.arange(1, len(cdf_x) + 1) / len(cdf_x)

    percentiles = np.percentile(data, [5, 25, 50, 75, 95])

    fig = go.Figure()

    # Histogram bar
    fig.add_trace(go.Bar(
        x=hist_x[:-1], y=hist_y,
        marker_color='darkblue',
        opacity=0.8,
        name='Histogram',
        width=(hist_x[1] - hist_x[0]),
        marker_line_width=0  # tanpa border
    ))

    # KDE line
    fig.add_trace(go.Scatter(
        x=kde_x, y=kde_y,
        mode='lines',
        line=dict(color='limegreen', width=2),
        name='KDE'
    ))

    # CDF line (sumbu kanan)
    fig.add_trace(go.Scatter(
        x=cdf_x, y=cdf_y,
        mode='lines',
        line=dict(color='gray', dash='dot'),
        name='CDF',
        yaxis='y2'
    ))

    # Background & layout styling
    fig.update_layout(
        title=f"Histogram: {log_column}",
        xaxis_title=log_column,
        yaxis=dict(
            title="Density",
            showgrid=True,
            gridcolor="black",
            gridwidth=0.4,
            tick0=0,
            dtick=0.1
        ),
        yaxis2=dict(
            title="CDF",
            overlaying='y',
            side='right',
            showgrid=True,
            gridcolor="black",
            gridwidth=0.3,
            tick0=0,
            dtick=0.2
        ),
        plot_bgcolor='rgba(230,230,230,1)',
        paper_bgcolor='rgba(230,230,210,1)',
        bargap=0.05,
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=60),
        height=500
    )

    # Tambahkan info persentil di bawah plot
    percentile_text = "<br>".join([
        f"P5: {percentiles[0]:.2f}",
        f"P25: {percentiles[1]:.2f}",
        f"P50: {percentiles[2]:.2f}",
        f"P75: {percentiles[3]:.2f}",
        f"P95: {percentiles[4]:.2f}"
    ])

    fig.add_annotation(
        text=percentile_text,
        xref="paper", yref="paper",
        x=0.01, y=-0.3,
        showarrow=False,
        align="left",
        font=dict(size=12, color="black")
    )

    return fig
