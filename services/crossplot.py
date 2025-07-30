from collections import Counter
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def calc_gradient(x1, y1, x2, y2):
    with np.errstate(divide='ignore', invalid='ignore'):
        grad = np.abs(y2 - y1) / np.abs(x2 - x1)
        grad = np.where(np.isinf(grad), np.nan, grad)
    return grad


def calc_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return (px, py)


def generate_crossplot(df, x_col, y_col, gr_ma, gr_sh, rho_ma, rho_sh, nphi_ma, nphi_sh, selected_intervals):
    # Filter berdasarkan marker jika ada
    if selected_intervals and 'MARKER' in df.columns:
        print(f"Filtering data for intervals: {selected_intervals}")
        df_filtered = df[df['MARKER'].isin(selected_intervals)].copy()
    else:
        df_filtered = df.copy()

    # Override kolom y jika GR norm tersedia dan y bukan RHOB
    if "GR_RAW_NORM" in df_filtered.columns and y_col != "RHOB":
        y_col = "GR_RAW_NORM"

    # Validasi kolom
    for col in [x_col, y_col]:
        if col not in df_filtered.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan dalam data.")

    # Tentukan color_col jika cocok NPHI vs RHOB
    color_col = None
    color_label = "Gamma Ray (API)"
    if x_col == "NPHI" and y_col == "RHOB":
        if "GR_RAW_NORM" in df_filtered.columns:
            color_col = "GR_RAW_NORM"
        elif "GR" in df_filtered.columns:
            color_col = "GR"

    # Bersihkan data
    if color_col and color_col in df_filtered.columns:
        df_clean = df_filtered[[x_col, y_col, color_col]].dropna()
    else:
        df_clean = df_filtered[[x_col, y_col]].dropna()
        df_clean["count"] = 1
        df_clean = df_clean.groupby([x_col, y_col]).count().reset_index()
        color_col = "count"

    if df_clean.empty:
        raise ValueError("Tidak ada data valid untuk crossplot.")

    # Scatter plot
    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        color=color_col,
        color_continuous_scale=[
            [0.0, "blue"],
            [0.25, "cyan"],
            [0.5, "yellow"],
            [0.75, "orange"],
            [1.0, "red"]
        ],
        labels={x_col: x_col, y_col: y_col, color_col: color_label},
        height=600,
    )

    # Auto QZ-WTR lines jika NPHI-RHOB
    if x_col == "NPHI" and y_col == "RHOB":
        x_quartz = -0.02
        y_quartz = 2.65
        x_water = 1.0
        y_water = 1.0

        prcnt_qz = 10
        prcnt_wtr = 1

        # hitung semua gradient QUARTZ (QZ) terhadap semua poin NPHI-RHOB
        df_clean.loc[df_clean.index, "GRAD_QZ"] = calc_gradient(
            x_quartz, y_quartz, df_clean["NPHI"], df_clean["RHOB"])
        df_QZ = df_clean[df_clean['GRAD_QZ'] > 0].sort_values(by='GRAD_QZ')
        # df_QZ = df_clean.sort_values(by='GRAD_QZ')
        df_QZ = df_QZ.iloc[int(prcnt_qz/100*len(df_QZ)):, :]
        x_line = np.array([x_quartz, df_QZ.iloc[0, :]['NPHI']])
        y_line = np.array([y_quartz, df_QZ.iloc[0, :]['RHOB']])

        # hitung semua gradient WATER (WTR) terhadap semua poin NPHI-RHOB
        df_clean.loc[df_clean.index, "GRAD_WTR"] = calc_gradient(
            x_water, y_water, df_clean["RHOB"], df_clean["NPHI"])
        df_WTR = df_clean[df_clean['GRAD_WTR'] > 0].sort_values(by='GRAD_WTR')
        # df_WTR = df_clean.sort_values(by='GRAD_WTR')
        df_WTR = df_WTR.iloc[int(prcnt_wtr/100*len(df_WTR)):, :]
        x_line_wtr = np.array([x_water, df_WTR.iloc[0, :]['NPHI']])
        y_line_wtr = np.array([y_water, df_WTR.iloc[0, :]['RHOB']])

        # setup dan konversi setiap titik persamaan garis linear (konversi ke REAL koordinat Cartesian)
        # semua sumbu y relatif terhadap y_quartz
        pq1 = (0.0, y_quartz - y_quartz)
        pq2 = (df_QZ.iloc[0, :]['NPHI'], (y_quartz - df_QZ.iloc[0, :]['RHOB']))
        pw1 = (df_WTR.iloc[0, :]['NPHI'],
               (y_quartz - df_WTR.iloc[0, :]['RHOB']))
        pw2 = (x_water, y_quartz - y_water)

        # cari perpotongan garis
        intersect = calc_intersection(pq1, pq2, pw1, pw2)

        if intersect:
            xx0 = intersect[0]
            yy0 = y_quartz - intersect[1]
            x_line_qz1 = np.array([x_quartz, xx0])
            y_line_qz1 = np.array([y_quartz, yy0])
            x_line_wtr1 = np.array([xx0, x_water])
            y_line_wtr1 = np.array([yy0, y_water])

            # 🛠 Perbaikan: Garis dan intersection seperti sebelumnya
            fig.add_trace(go.Scatter(
                x=x_line_qz1,
                y=y_line_qz1,
                mode='lines',
                line=dict(color='red', width=2),
                name='QZ Line'
            ))
            fig.add_trace(go.Scatter(
                x=x_line_wtr1,
                y=y_line_wtr1,
                mode='lines',
                line=dict(color='red', width=2),
                name='WTR Line'
            ))
            fig.add_trace(go.Scatter(
                x=[xx0],
                y=[yy0],
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle'),
                name='Intersection'
            ))
            fig.add_trace(go.Scatter(
                x=[x_quartz, x_water],
                y=[y_quartz, y_water],
                mode='markers',
                marker=dict(color='black', size=5, symbol='cross'),
                name='Quartz & Water'
            ))

            fig.add_shape(
                type="line",
                x0=1, y0=1,
                x1=-0.02, y1=2.65,
                xref='x', yref='y',
                line=dict(color="red", width=2, dash="solid"),
                layer='above'
            )

            # Layout
            fig.update_layout(
                title=f"Crossplot {x_col} vs {y_col}",
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
                    range=[3, 1],  # 🛠 Membalik sumbu RHOB
                    dtick=0.2,
                    showgrid=True,
                    gridcolor="black",
                    gridwidth=0.2
                ),
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=40),
                coloraxis_colorbar=dict(
                    title=dict(text=color_label, side='bottom'),
                    orientation='h',
                    y=-0.3,
                    x=0.5,
                    xanchor='center',
                    len=1,
                ),
            )

    elif x_col == "NPHI" and (y_col == "GR" or y_col == "GR_RAW_NORM"):
        # tentukan titik koordinat Quartz dan Water (relatif terhada visualisasi, bukan real x dan y)
        y_max = df_clean[y_col].max()
        yaxis_range = [0, math.ceil(y_max / 20) * 20]
        yaxis_dtick = 20

        x_quartz = -0.02
        y_quartz = y_max
        x_water = 1.0
        y_water = 0.0

        # tentukan threshold/percentile gradient yang akan dibuang
        prcnt_qz = 5
        prcnt_wtr = 5

        # Percentile Quartz = 5% ke BAWAH
        # Percentile Water = 5% ke KANAN

        count = Counter(df_clean[y_col])
        freq = [count[v] for v in df_clean[y_col]]
        color_label = "Frekuensi"

        df_clean["COLOR"] = freq

        # hitung semua gradient QUARTZ (QZ) terhadap semua poin NPHI-RHOB
        df_clean.loc[df_clean.index, "GRAD_QZ"] = calc_gradient(
            x_quartz, y_quartz, df_clean[x_col], df_clean[y_col])
        df_QZ = df_clean[df_clean['GRAD_QZ'] > 0].sort_values(by='GRAD_QZ')
        # df_QZ = df_clean.sort_values(by='GRAD_QZ')
        df_QZ = df_QZ.iloc[int(prcnt_qz/100*len(df_QZ)):, :]
        x_line = np.array([x_quartz, df_QZ.iloc[0, :][x_col]])
        y_line = np.array([y_quartz, df_QZ.iloc[0, :][y_col]])

        # hitung semua gradient WATER (WTR) terhadap semua poin NPHI-RHOB
        df_clean.loc[df_clean.index, "GRAD_WTR"] = calc_gradient(
            x_water, y_water, df_clean[y_col], df_clean[x_col])
        df_WTR = df_clean[df_clean['GRAD_WTR'] > 0].sort_values(by='GRAD_WTR')
        # df_WTR = df_clean.sort_values(by='GRAD_WTR')
        df_WTR = df_WTR.iloc[int(prcnt_wtr/100*len(df_WTR)):, :]
        x_line_wtr = np.array([x_water, df_WTR.iloc[0, :][x_col]])
        y_line_wtr = np.array([y_water, df_WTR.iloc[0, :][y_col]])

        # setup dan konversi setiap titik persamaan garis linear (konversi ke REAL koordinat Cartesian)
        # semua sumbu y relatif terhadap y_quartz
        pq1 = (0.0, y_quartz - y_quartz)
        pq2 = (df_QZ.iloc[0, :][x_col], (y_quartz - df_QZ.iloc[0, :][y_col]))
        pw1 = (df_WTR.iloc[0, :][x_col],
               (y_quartz - df_WTR.iloc[0, :][y_col]))
        pw2 = (x_water, y_quartz - y_water)

        # cari perpotongan garis
        intersect = calc_intersection(pq1, pq2, pw1, pw2)

        if intersect:
            # konversi perpotongan garis ke koordinat visualisasi
            xx0 = intersect[0]
            yy0 = y_quartz - intersect[1]
            x_line_qz1 = np.array([x_quartz, xx0])
            y_line_qz1 = np.array([y_quartz, yy0])
            x_line_wtr1 = np.array([xx0, x_water])
            y_line_wtr1 = np.array([yy0, y_water])

            # fig.add_trace(go.Scatter(
            #     x=x_line_qz1,
            #     y=y_line_qz1,
            #     mode='lines',
            #     line=dict(color='red', width=2),
            #     name='QZ Line'
            # ))
            # fig.add_trace(go.Scatter(
            #     x=x_line_wtr1,
            #     y=y_line_wtr1,
            #     mode='lines',
            #     line=dict(color='red', width=2),
            #     name='WTR Line'
            # ))
            # fig.add_trace(go.Scatter(
            #     x=[xx0],
            #     y=[yy0],
            #     mode='markers',
            #     marker=dict(color='red', size=10, symbol='circle'),
            #     name='Intersection'
            # ))

            fig.add_shape(
                type="line",
                x0=1, y0=0,
                x1=-0.02, y1=gr_ma,
                xref='x', yref='y',
                line=dict(color="black", width=2, dash="solid"),
                layer='above'
            )
            fig.add_shape(
                type="line",
                x0=-0.02, y0=gr_ma,
                x1=0.4, y1=220,
                xref='x', yref='y',
                line=dict(color="black", width=2, dash="solid"),
                layer='above'
            )
            fig.add_shape(
                type="line",
                x0=0.4, y0=220,
                x1=1, y1=0,
                xref='x', yref='y',
                line=dict(color="black", width=2, dash="solid"),
                layer='above'
            )

            # Layout
            fig.update_layout(
                title=f"Crossplot {x_col} vs {y_col}",
                xaxis=dict(
                    title='NPHI (V/V)',
                    range=[-0.1, 1],
                    dtick=0.1,
                    showgrid=True,
                    gridcolor="black",
                    gridwidth=0.2
                ),
                yaxis=dict(
                    title='GR (API)',
                    range=yaxis_range,
                    dtick=yaxis_dtick,
                    showgrid=True,
                    gridcolor="black",
                    gridwidth=0.2
                ),
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=40),
                coloraxis_colorbar=dict(
                    title=dict(text=color_label, side='bottom'),
                    orientation='h',
                    y=-0.3,
                    x=0.5,
                    xanchor='center',
                    len=1,
                ),
            )

    return fig
