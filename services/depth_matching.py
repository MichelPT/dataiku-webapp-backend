# services/depth_matching.py

import sys
import os
import lasio
import numpy as np
import pandas as pd
import time
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Konfigurasi Path untuk Impor Modul 'cow' ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    utils_path = os.path.join(base_dir, 'utils')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    import cow_fixed as cow
    logging.info(
        "Modul 'cow_fixed' berhasil diimpor sebagai 'cow' dari folder 'utils'.")
except ImportError as e:
    logging.error(
        "GAGAL mengimpor modul 'cow_fixed'. Pastikan 'cow_fixed.py' ada di dalam folder 'utils'.")
    raise e

# --- FUNGSI LOGIKA UTAMA (DIUBAH) ---


def depth_matching(ref_las_path: str, lwd_las_path: str, ref_log_curve: str, lwd_log_curve: str, num_chunks: int = 15, slack: int = 20):
    logging.info(f"--- Memulai Proses Depth Matching (Logika Baru) ---")

    # 1. Baca file LAS dan simpan DataFrame asli
    ref_las = lasio.read(ref_las_path)
    lwd_las = lasio.read(lwd_las_path)
    ref_df_orig = ref_las.df().reset_index().rename(columns={'DEPT': 'DEPTH'})
    lwd_df_orig = lwd_las.df().reset_index().rename(columns={'DEPT': 'DEPTH'})

    # Validasi kolom
    if 'DEPTH' not in ref_df_orig.columns or ref_log_curve not in ref_df_orig.columns:
        raise ValueError(
            f"Kolom '{ref_log_curve}' atau 'DEPTH' tidak ada di file Referensi.")
    if 'DEPTH' not in lwd_df_orig.columns or lwd_log_curve not in lwd_df_orig.columns:
        raise ValueError(
            f"Kolom '{lwd_log_curve}' atau 'DEPTH' tidak ada di file LWD.")

    # --- LOGIKA FINAL: Menyelaraskan dan Mengisi Data Sesuai Referensi ---
    logging.info(
        "Membangun DataFrame yang selaras berdasarkan rentang kedalaman referensi...")

    # Set DEPTH sebagai index untuk semua operasi
    ref_df = ref_df_orig.set_index('DEPTH')
    lwd_df = lwd_df_orig.set_index('DEPTH')

    # 1. Buat DataFrame baru yang terpusat, menggunakan index dari referensi
    # Ini secara otomatis melakukan TRIM dan menyisakan NaN untuk data LWD yang perlu diisi
    aligned_df = pd.DataFrame(index=ref_df.index)
    aligned_df['REF'] = ref_df[ref_log_curve]
    aligned_df['LWD'] = lwd_df[lwd_log_curve].reindex(ref_df.index)

    # 2. Lakukan pembersihan data pada DataFrame yang sudah selaras
    aligned_df['REF'] = pd.to_numeric(aligned_df['REF'], errors='coerce')
    aligned_df['LWD'] = pd.to_numeric(aligned_df['LWD'], errors='coerce')
    aligned_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Tangani nilai NaN sesuai aturan
    # Pertama, buang baris di mana data REFERENSI itu sendiri tidak valid
    aligned_df.dropna(subset=['REF'], inplace=True)

    # Kedua, ISI data LWD yang hilang menggunakan backward fill lalu forward fill
    logging.info(
        "Mengisi nilai LWD yang hilang dengan metode bfill dan ffill...")
    aligned_df['LWD'].fillna(method='bfill', inplace=True)
    aligned_df['LWD'].fillna(method='ffill', inplace=True)

    # Terakhir, sebagai jaring pengaman, buang baris jika LWD masih NaN (artinya tidak ada data LWD sama sekali)
    final_df = aligned_df.dropna()

    if final_df.empty:
        raise ValueError(
            "Tidak ada data valid yang tersisa setelah proses alignment dan fill.")

    # 4. Ekstrak sinyal final. Panjangnya DIJAMIN SAMA.
    ref_signal = final_df['REF'].values
    lwd_signal = final_df['LWD'].values
    # -----------------------------------------------------------------------------

    logging.info(
        f"Sinyal siap untuk COW dengan panjang yang identik: {len(ref_signal)}")

    # 5. Jalankan algoritma COW
    aligner = cow.COW(lwd_signal, ref_signal, num_chunks, slack)
    aligned_lwd_signal, _ = aligner.warp_sample_to_target()

    # 6. Buat DataFrame hasil akhir
    final_result_df = pd.DataFrame({
        'DEPTH': final_df.index,
        'LOG_REF': final_df['REF'].values,
        'LOG_ALIGNED_LWD': aligned_lwd_signal
    })

    return ref_df_orig, lwd_df_orig, final_result_df


def plot_depth_matching_results(ref_df: pd.DataFrame, lwd_df: pd.DataFrame, final_df: pd.DataFrame, ref_log_curve: str, lwd_log_curve: str):
    """
    Membuat visualisasi Plotly menggunakan nama kurva yang spesifik.
    """
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Before Alignment", "After Alignment")
    )

    # Track 1: Sebelum Alignment (menggunakan nama kurva spesifik)
    fig.add_trace(go.Scatter(
        x=ref_df[ref_log_curve], y=ref_df['DEPTH'], name=f'WL ({ref_log_curve})', line=dict(color='black')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lwd_df[lwd_log_curve], y=lwd_df['DEPTH'], name=f'LWD ({lwd_log_curve})', line=dict(color='red', dash='dash')
    ), row=1, col=1)

    # Track 2: Sesudah Alignment (menggunakan kolom generik dari final_df)
    fig.add_trace(go.Scatter(
        x=final_df['LOG_REF'], y=final_df['DEPTH'], name=f'WL ({ref_log_curve})', line=dict(color='black'), showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=final_df['LOG_ALIGNED_LWD'], y=final_df['DEPTH'], name=f'LWD Aligned ({lwd_log_curve})', line=dict(color='blue')
    ), row=1, col=2)

    # Update Layout
    fig.update_layout(
        title_text=f'Depth Matching Result: {ref_log_curve} vs {lwd_log_curve}',
        height=800, yaxis_title='DEPTH (m)',
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(autorange="reversed")

    return fig
