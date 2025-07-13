import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import math


def generate_crossplot(df: pd.DataFrame, x_col: str, y_col: str, gr_ma: float, gr_sh: float):
    df_clean = df[[x_col, y_col]].dropna()

    if df_clean.empty:
        raise ValueError("Tidak ada data valid untuk crossplot.")

    # Tambahkan kolom warna
    if x_col == "NPHI" and y_col == "RHOB":
        if "GR" not in df.columns:
            raise ValueError(
                "Kolom GR diperlukan untuk plotting ini.")
        df_clean = df_clean.copy()
        df_clean["COLOR"] = df.loc[df_clean.index, "GR"]
        color_label = "GR (API)"
        color_continuous_title = "Gamma Ray (API)"
        show_shapes = 1

        yaxis_range = [3, 1]
        yaxis_dtick = 0.2

    else:
        count = Counter(df_clean[x_col])
        freq = [min(count[x], 5) for x in df_clean[x_col]]
        df_clean["COLOR"] = freq
        color_label = "Frekuensi"
        color_continuous_title = "Frekuensi"
        show_shapes = 2

        if y_col == "GR":
            y_min = df_clean["GR"].min()
            y_max = df_clean["GR"].max()
            yaxis_range = [
                0,
                math.ceil(y_max / 20) * 20
            ]
            yaxis_dtick = 20
        else:
            yaxis_range = None
            yaxis_dtick = None

    range_color = [1, 5] if x_col == "NPHI" and y_col == "GR" else None
    labels = {'NPHI': 'NPHI (V/V)', 'RHOB': 'RHOB (g/cc)', 'GR': 'Gamma Ray (API)'} if x_col == "NPHI" and y_col == "RHOB" else {
        'NPHI': 'NPHI (V/V)', 'GR': 'Gamma Ray (API)', "COLOR": color_label
    }

    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        color="COLOR",
        color_continuous_scale=[
            [0.0, "blue"],
            [0.25, "cyan"],
            [0.5, "yellow"],
            [0.75, "orange"],
            [1.0, "red"]
        ],
        labels=labels,
        range_color=range_color,
        title=f"Crossplot {x_col} vs {y_col}<br><sub>Warna berdasarkan {color_label}</sub>",
        width=700,
        height=700
    )

    fig.update_traces(
        marker=dict(size=6, line=dict(width=0.5, color='black')),
        opacity=0.7
    )

    if show_shapes == 1:
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                title='NPHI (V/V)',
                range=[-0.1, 1],
                dtick=0.1,
                showgrid=True,
                gridcolor="black",
                gridwidth=0.2
            ),
            yaxis=dict(
                title='RHOB (g/cc)',
                range=[3, 1],
                dtick=0.2,
                showgrid=True,
                gridcolor="black",
                gridwidth=0.2
            ),
            margin=dict(l=20, r=20, t=60, b=40),
            coloraxis_colorbar=dict(
                title=dict(text='GR (API)', side='bottom'),
                orientation='h',
                y=-0.3,
                x=0.5,
                xanchor='center',
                len=1
            )
        )

        fig.add_shape(
            type="line",
            x0=-0.02, x1=1,
            y0=2.65, y1=1,
            line=dict(color="black", width=1, dash="solid"),
        )
        fig.add_shape(
            type="line",
            x0=0.43, x1=1,
            y0=2.4, y1=1,
            line=dict(color="black", width=1, dash="solid"),
        )
        fig.add_shape(
            type="line",
            x0=-0.02, x1=0.43,
            y0=2.65, y1=2.4,
            line=dict(color="black", width=1, dash="solid"),
        )
    elif x_col == "NPHI" and y_col == "GR":
        fig.update_layout(
            xaxis=dict(
                title='NPHI (V/V)',
                range=[-0.1, 1],
                dtick=0.1,
                showgrid=True,
                gridcolor="black",
                gridwidth=0.2,
            ),
            yaxis=dict(
                title='GR (API)',
                range=yaxis_range,
                dtick=20,
                showgrid=True,
                gridcolor="black",
                gridwidth=0.2,
            ),
            margin=dict(l=20, r=20, t=60, b=40),
            coloraxis_colorbar=dict(
                title=dict(text=color_continuous_title, side='bottom'),
                orientation='h',
                y=-0.3,
                x=0.5,
                xanchor='center',
                len=1,
                tickvals=[1, 2, 3, 4, 5]
            )
        )
        fig.add_shape(
            type="line",
            x0=1, y0=0,
            x1=-0.02, y1=gr_ma,  # y1 digantikan input user GR_MA kalau ada, jika tidak ada tetap 30
            xref='x', yref='y',
            line=dict(color="black", width=2, dash="solid"),
            layer='above'
        )
        fig.add_shape(
            type="line",
            x0=-0.02, y0=gr_ma,  # y0 digantikan input user GR_MA kalau ada, jika tidak ada tetap 30
            x1=0.4, y1=gr_sh,  # y1 digantikan input user GR_SH kalau ada, jika tidak ada tetap 120
            xref='x', yref='y',
            line=dict(color="black", width=2, dash="solid"),
            layer='above'
        )
        fig.add_shape(
            type="line",
            x0=0.4, y0=gr_sh,  # y0 digantikan input user GR_SH kalau ada, jika tidak ada tetap 120
            x1=1, y1=0,
            xref='x', yref='y',
            line=dict(color="black", width=2, dash="solid"),
            layer='above'
        )

    return fig
