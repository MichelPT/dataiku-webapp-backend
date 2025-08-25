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
    logging.info("Modul 'cow' berhasil diimpor dari folder 'utils'.")
except ImportError as e:
    logging.error(
        "GAGAL mengimpor modul 'cow'. Pastikan 'cow.py' ada di folder 'utils'.")
    raise e

# --- FUNGSI HELPER: Normalisasi Sinyal ---


def normalize(series: np.ndarray) -> np.ndarray:
    """Mengubah skala sinyal ke mean=0 dan std_dev=1."""
    mean = np.mean(series)
    std_dev = np.std(series)
    if std_dev == 0:
        return np.zeros_like(series)
    return (series - mean) / std_dev

# --- FUNGSI LOGIKA UTAMA (PALING STABIL) ---


def depth_matching(ref_las_path: str, lwd_las_path: str, ref_log_curve: str, lwd_log_curve: str, num_chunks: int = 15, slack: int = 30):
    logging.info(f"--- Memulai Proses Depth Matching (dengan Normalisasi) ---")

    # 1. Baca dan validasi data
    ref_las = lasio.read(ref_las_path)
    lwd_las = lasio.read(lwd_las_path)
    ref_df_orig = ref_las.df().reset_index().rename(columns={'DEPT': 'DEPTH'})
    lwd_df_orig = lwd_las.df().reset_index().rename(columns={'DEPT': 'DEPTH'})

    if 'DEPTH' not in ref_df_orig.columns or ref_log_curve not in ref_df_orig.columns:
        raise ValueError(
            f"Kolom '{ref_log_curve}' atau 'DEPTH' tidak ada di file Referensi.")
    if 'DEPTH' not in lwd_df_orig.columns or lwd_log_curve not in lwd_df_orig.columns:
        raise ValueError(
            f"Kolom '{lwd_log_curve}' atau 'DEPTH' tidak ada di file LWD.")

    # 2. Selaraskan, bersihkan, dan isi data
    ref_df = ref_df_orig.set_index('DEPTH')
    lwd_df = lwd_df_orig.set_index('DEPTH')

    # Buat DataFrame terpusat berdasarkan index referensi (otomatis trim)
    aligned_df = pd.DataFrame(index=ref_df.index)
    aligned_df['REF'] = ref_df[ref_log_curve]
    aligned_df['LWD'] = lwd_df[lwd_log_curve].reindex(ref_df.index)

    # Bersihkan data
    aligned_df['REF'] = pd.to_numeric(aligned_df['REF'], errors='coerce')
    aligned_df['LWD'] = pd.to_numeric(aligned_df['LWD'], errors='coerce')
    aligned_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Tangani nilai NaN
    aligned_df.dropna(subset=['REF'], inplace=True)
    aligned_df['LWD'] = aligned_df['LWD'].bfill().ffill()
    final_df = aligned_df.dropna()

    if final_df.empty:
        raise ValueError(
            "Tidak ada data valid yang tersisa setelah proses alignment dan fill.")

    ref_signal_orig = final_df['REF'].values
    lwd_signal_orig = final_df['LWD'].values

    # 3. Normalisasi Sinyal (Langkah Kunci)
    logging.info("Normalisasi sinyal referensi dan LWD...")
    ref_signal_norm = normalize(ref_signal_orig)
    lwd_signal_norm = normalize(lwd_signal_orig)

    # Simpan parameter LWD asli untuk denormalisasi
    lwd_mean = np.mean(lwd_signal_orig)
    lwd_std = np.std(lwd_signal_orig)

    # 4. Jalankan algoritma COW pada data yang sudah dinormalisasi
    logging.info(
        f"Sinyal ternormalisasi siap untuk COW dengan panjang: {len(ref_signal_norm)}")
    aligner = cow.COW(lwd_signal_norm, ref_signal_norm, num_chunks, slack)
    aligned_lwd_signal_norm, _ = aligner.warp_sample_to_target()

    # 5. Denormalisasi Hasil (mengembalikan ke skala asli)
    if lwd_std > 0:
        aligned_lwd_signal = (aligned_lwd_signal_norm * lwd_std) + lwd_mean
    else:
        aligned_lwd_signal = aligned_lwd_signal_norm + lwd_mean

    # 6. Buat DataFrame hasil akhir
    final_result_df = pd.DataFrame({
        'DEPTH': final_df.index,
        'LOG_REF': ref_signal_orig,
        'LOG_ALIGNED_LWD': aligned_lwd_signal
    })

    return ref_df_orig, lwd_df_orig, final_result_df

# --- FUNGSI PLOTTING ---


def plot_depth_matching_results(ref_df: pd.DataFrame, lwd_df: pd.DataFrame, final_df: pd.DataFrame, ref_log_curve: str, lwd_log_curve: str):
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Before Alignment", "After Alignment")
    )

    # Track 1: Sebelum Alignment
    fig.add_trace(go.Scatter(
        x=ref_df[ref_log_curve], y=ref_df['DEPTH'], name=f'WL ({ref_log_curve})', line=dict(color='black')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lwd_df[lwd_log_curve], y=lwd_df['DEPTH'], name=f'LWD ({lwd_log_curve})', line=dict(color='red', dash='dash')
    ), row=1, col=1)

    # Track 2: Sesudah Alignment
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
