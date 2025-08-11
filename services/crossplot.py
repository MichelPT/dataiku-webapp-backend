import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import math

# Pastikan Anda mengimpor fungsi terpusat yang telah dibuat sebelumnya
# Asumsikan file ini bernama petrophysics_calculations.py
from services.autoplot import calculate_gr_ma_sh_from_nphi_rhob, calculate_nphi_rhob_intersection


def generate_crossplot(df, x_col, y_col, gr_ma, gr_sh, rho_ma, rho_sh, nphi_ma, nphi_sh, prcnt_qz, prcnt_wtr, selected_intervals, nbins=25):
    """
    Membuat visualisasi crossplot untuk data log sumur.
    Kini menggunakan fungsi terpusat untuk perhitungan intersection.
    """
    # Filter berdasarkan marker jika ada
    if selected_intervals and 'MARKER' in df.columns:
        print(f"Filtering data for intervals: {selected_intervals}")
        df_filtered = df[df['MARKER'].isin(selected_intervals)].copy()
    else:
        df_filtered = df.copy()

    # Validasi kolom
    for col in [x_col, y_col]:
        if col not in df_filtered.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan dalam data.")

    # Tentukan kolom warna awal jika NPHI vs RHOB
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

    if df_clean.empty:
        raise ValueError(
            "Tidak ada data valid untuk crossplot setelah dibersihkan.")

    # Inisialisasi figure
    fig = go.Figure()

    # --- Blok 1: Logika untuk Plot NPHI vs RHOB ---
    if x_col == "NPHI" and y_col == "RHOB":
        # Urutkan data berdasarkan warna untuk layering plot
        df_clean = df_clean.sort_values(by=color_col, ascending=True)

        # Buat scatter plot dasar
        fig = px.scatter(
            df_clean,
            x=x_col,
            y=y_col,
            color=color_col,
            color_continuous_scale=[
                [0.0, "blue"], [0.25, "cyan"], [0.5, "yellow"], [
                    0.75, "orange"], [1.0, "red"]
            ],
            labels={x_col: x_col, y_col: y_col, color_col: color_label},
            height=600,
        )

        # --- MENGGUNAKAN FUNGSI TERPUSAT ---
        try:
            # Panggil fungsi terpusat untuk mendapatkan koordinat intersection
            intersection_coords = calculate_nphi_rhob_intersection(
                df_filtered, prcnt_qz, prcnt_wtr)

            xx0 = intersection_coords["nphi_sh"]
            yy0 = intersection_coords["rhob_sh"]

            # Gunakan koordinat untuk menggambar garis dan titik pada plot
            x_quartz, y_quartz = -0.02, 2.65
            x_water, y_water = 1.0, 1.0

            x_line_qz1 = np.array([x_quartz, xx0])
            y_line_qz1 = np.array([y_quartz, yy0])
            x_line_wtr1 = np.array([xx0, x_water])
            y_line_wtr1 = np.array([yy0, y_water])

            # Tambahkan shapes dan traces ke 'fig'
            fig.add_trace(go.Scatter(x=[x_quartz, x_water], y=[y_quartz, y_water], mode='markers', marker=dict(
                color='black', size=5, symbol='cross'), name='Quartz & Water'))
            fig.add_shape(type="line", x0=x_quartz, y0=y_quartz, x1=x_water,
                          y1=y_water, line=dict(color="red", width=2), layer='above')
            fig.add_shape(type="line", x0=x_line_qz1[0], y0=y_line_qz1[0], x1=x_line_qz1[-1],
                          y1=y_line_qz1[-1], line=dict(color="red", width=2), layer='above')
            fig.add_shape(type="line", x0=x_line_wtr1[0], y0=y_line_wtr1[0], x1=x_line_wtr1[-1],
                          y1=y_line_wtr1[-1], line=dict(color="red", width=2), layer='above')
            fig.add_trace(go.Scatter(x=[xx0], y=[yy0], mode='markers', marker=dict(
                color='red', size=2, symbol='circle'), name='Intersection'))

            # Lingkaran di sekitar titik intersection
            radiusX, radiusY = 0.01, 0.02
            fig.add_shape(type="circle", xref="x", yref="y", x0=xx0-radiusX, y0=yy0-radiusY,
                          x1=xx0+radiusX, y1=yy0+radiusY, line_color="red", fillcolor="red", layer='above')

        except ValueError as e:
            # Jika perhitungan intersection gagal, plot tetap ditampilkan tanpa garis-garis ini
            print(f"Peringatan saat menghitung intersection untuk plot: {e}")

        # Update layout untuk NPHI vs RHOB
        fig.update_layout(
            title=f"Crossplot {x_col} vs {y_col}",
            xaxis=dict(title='NPHI (V/V)', range=[-0.1, 1], dtick=0.1,
                       showgrid=True, gridcolor="black", gridwidth=0.2),
            yaxis=dict(title='RHOB (g/cc)', range=[3, 1], dtick=0.2,
                       showgrid=True, gridcolor="black", gridwidth=0.2),
            plot_bgcolor='white', margin=dict(l=20, r=20, t=60, b=40),
            coloraxis_colorbar=dict(title=dict(
                text=color_label, side='bottom'), orientation='h', y=-0.3, x=0.5, xanchor='center', len=1),
        )

    # --- Blok 2: Logika untuk Plot NPHI vs GR ---
    elif x_col == "NPHI" and (y_col == "GR" or y_col == "GR_RAW_NORM"):
        count = Counter(df_clean[y_col])
        freq = [count[v] for v in df_clean[y_col]]
        df_clean["COLOR"] = freq
        color_label = "Frekuensi"

        df_clean = df_clean.sort_values(by='COLOR', ascending=True)

        fig = px.scatter(
            df_clean, x=x_col, y=y_col, color="COLOR",
            color_continuous_scale=[
                [0.0, "blue"], [0.25, "cyan"], [0.5, "yellow"], [
                    0.75, "orange"], [1.0, "red"]
            ],
            labels={x_col: "NPHI (V/V)", y_col: "GR (API)",
                    "COLOR": color_label},
            height=600,
        )

        y_max = df_clean[y_col].max()
        yaxis_range = [0, y_max + 20]
        yaxis_dtick = 20

        gr_value = calculate_gr_ma_sh_from_nphi_rhob(
            df_filtered, prcnt_qz, prcnt_wtr)
        gr_ma = gr_value['gr_ma']
        gr_sh = gr_value['gr_sh']

        intersection_coords = calculate_nphi_rhob_intersection(
            df_filtered, prcnt_qz, prcnt_wtr)

        xx0 = intersection_coords["nphi_sh"]

        # Gunakan koordinat untuk menggambar garis dan titik pada plot
        nphi_ma = -0.02
        nphi_sh = xx0

        fig.add_shape(type="line", x0=1, y0=0, x1=nphi_ma, y1=gr_ma, xref='x',
                      yref='y', line=dict(color="red", width=2, dash="solid"), layer='above')
        fig.add_shape(type="line", x0=nphi_ma, y0=gr_ma, x1=nphi_sh, y1=gr_sh, xref='x',
                      yref='y', line=dict(color="red", width=2, dash="solid"), layer='above')
        fig.add_shape(type="line", x0=nphi_sh, y0=gr_sh, x1=1, y1=0, xref='x', yref='y', line=dict(
            color="red", width=2, dash="solid"), layer='above')

        min_freq = df_clean['COLOR'].min()
        max_freq = df_clean['COLOR'].max()
        tick_step = max(1, round((max_freq - min_freq) / 5)
                        ) if max_freq > min_freq else 1

        fig.update_layout(
            title=f"Crossplot {x_col} vs {y_col}",
            xaxis=dict(title='NPHI (V/V)', range=[-0.1, 1], dtick=0.1,
                       showgrid=True, gridcolor="black", gridwidth=0.2),
            yaxis=dict(title='GR (API)', range=yaxis_range, dtick=yaxis_dtick,
                       showgrid=True, gridcolor="black", gridwidth=0.2),
            plot_bgcolor='white', margin=dict(l=20, r=20, t=60, b=40),
            coloraxis_colorbar=dict(title=dict(text=color_label, side='bottom'),
                                    orientation='h', y=-0.3, x=0.5, xanchor='center', len=1, dtick=tick_step),
        )

    elif x_col == "RHOB" and (y_col == "GR" or y_col == "GR_RAW_NORM"):
        count = Counter(df_clean[y_col])
        freq = [count[v] for v in df_clean[y_col]]
        df_clean["COLOR"] = freq
        color_label = "Frekuensi"

        df_clean = df_clean.sort_values(by='COLOR', ascending=True)

        fig = px.scatter(
            df_clean, x=x_col, y=y_col, color="COLOR",
            color_continuous_scale=[
                [0.0, "blue"], [0.25, "cyan"], [0.5, "yellow"], [
                    0.75, "orange"], [1.0, "red"]
            ],
            labels={x_col: "RHOB (G/C3)", y_col: "GR (API)",
                    "COLOR": color_label},
            height=600,
        )

        y_max = df_clean[y_col].max()
        yaxis_range = [0, y_max + 20]
        yaxis_dtick = 20

        gr_value = calculate_gr_ma_sh_from_nphi_rhob(
            df_filtered, prcnt_qz, prcnt_wtr)
        gr_ma = gr_value['gr_ma']
        gr_sh = gr_value['gr_sh']

        intersection_coords = calculate_nphi_rhob_intersection(
            df_filtered, prcnt_qz, prcnt_wtr)

        yy0 = intersection_coords["rhob_sh"]

        # Gunakan koordinat untuk menggambar garis dan titik pada plot
        rhob_ma = 2.65
        rhob_sh = yy0

        fig.add_shape(type="line", x0=1, y0=0, x1=rhob_ma, y1=gr_ma, xref='x',
                      yref='y', line=dict(color="red", width=2, dash="solid"), layer='above')
        fig.add_shape(type="line", x0=rhob_ma, y0=gr_ma, x1=rhob_sh, y1=gr_sh, xref='x',
                      yref='y', line=dict(color="red", width=2, dash="solid"), layer='above')
        fig.add_shape(type="line", x0=rhob_sh, y0=gr_sh, x1=1, y1=0, xref='x', yref='y', line=dict(
            color="red", width=2, dash="solid"), layer='above')

        min_freq = df_clean['COLOR'].min()
        max_freq = df_clean['COLOR'].max()
        tick_step = max(1, round((max_freq - min_freq) / 5)
                        ) if max_freq > min_freq else 1

        fig.update_layout(
            title=f"Crossplot {x_col} vs {y_col}",
            xaxis=dict(title='RHOB (G/C3)', range=[1, 3], dtick=0.1,
                       showgrid=True, gridcolor="black", gridwidth=0.2),
            yaxis=dict(title='GR (API)', range=yaxis_range, dtick=yaxis_dtick,
                       showgrid=True, gridcolor="black", gridwidth=0.2),
            plot_bgcolor='white', margin=dict(l=20, r=20, t=60, b=40),
            coloraxis_colorbar=dict(title=dict(text=color_label, side='bottom'),
                                    orientation='h', y=-0.3, x=0.5, xanchor='center', len=1, dtick=tick_step),
        )

    # --- Blok 3: Fallback untuk Plot Lainnya ---
    else:
        df_clean = df_filtered[[x_col, y_col]].dropna()
        if df_clean.empty:
            raise ValueError(
                f"Tidak ada data valid untuk crossplot {x_col} vs {y_col}.")

        color_label = "Point Count"

        # A. Hitung histogram 2D secara manual dengan NumPy
        counts, x_edges, y_edges = np.histogram2d(
            df_clean[x_col],
            df_clean[y_col],
            bins=nbins
        )

        # B. Ganti semua nilai 0 dengan 'nan' agar tidak digambar
        counts[counts == 0] = np.nan

        # C. Hitung titik tengah bin untuk sumbu plot heatmap
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # D. Tambahkan trace go.Heatmap
        fig.add_trace(go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=counts.T,  # Matriks 'counts' perlu di-transpose (.T)
            colorscale='Jet',
            colorbar=dict(title=color_label, orientation='h',
                          y=-0.2, x=0.5, xanchor='center', len=1)
        ))

        # E. Jika ini plot NPHI vs GR, tambahkan garis overlay spesifiknya
        if x_col == "NPHI" and (y_col == "GR" or y_col == "GR_RAW_NORM"):
            fig.add_shape(type="line", x0=1, y0=0, x1=-0.02, y1=gr_ma,
                          line=dict(color="red", width=2, dash="solid"))
            fig.add_shape(type="line", x0=-0.02, y0=gr_ma, x1=0.4,
                          y1=gr_sh, line=dict(color="red", width=2, dash="solid"))
            fig.add_shape(type="line", x0=0.4, y0=gr_sh, x1=1,
                          y1=0, line=dict(color="red", width=2, dash="solid"))

        # F. Atur layout akhir secara dinamis
        fig.update_layout(
            title=f"Crossplot {x_col} vs {y_col}",
            plot_bgcolor='white', margin=dict(l=20, r=20, t=60, b=40),
            xaxis=dict(title=x_col, showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(title=y_col, showgrid=True, gridcolor="lightgrey"),
            showlegend=False
        )

    return fig
