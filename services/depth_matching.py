import sys
import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def import_cow():
    """
    Mengimpor library COW dari path yang spesifik.
    Pastikan path ini benar di lingkungan server Anda.
    """
    cow_path = "utils/cow_fixed.py"

    if not os.path.exists(cow_path):
        raise ImportError(
            f"Tidak dapat menemukan file 'cow_fixed.py' di: {cow_path}")

    try:
        spec = importlib.util.spec_from_file_location(
            "cow_fixed", str(cow_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cow_fixed"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        raise ImportError(f"Gagal mengimpor 'cow_fixed.py': {e}")


cow = import_cow()


def run_depth_matching(ref_csv_path: str, lwd_csv_path: str, ref_log_curve: str, lwd_log_curve: str, nbFrames: int, slack: int):
    """
    Fungsi utama untuk menjalankan proses depth matching dari file CSV.
    """
    try:
        # 1. Baca file CSV
        df_wl_full = pd.read_csv(ref_csv_path)
        df_lwd_full = pd.read_csv(lwd_csv_path)

        # Ganti nama 'DEPT' ke 'DEPTH' jika ada, tanpa mengubah DataFrame asli di memori
        if 'DEPT' in df_wl_full.columns:
            df_wl_full = df_wl_full.rename(columns={'DEPT': 'DEPTH'})
        if 'DEPT' in df_lwd_full.columns:
            df_lwd_full = df_lwd_full.rename(columns={'DEPT': 'DEPTH'})

        # 2. Validasi dan siapkan DataFrame
        if 'DEPTH' not in df_wl_full.columns or ref_log_curve not in df_wl_full.columns:
            raise KeyError(
                f"Kolom 'DEPTH' atau '{ref_log_curve}' tidak ditemukan di {os.path.basename(ref_csv_path)}")
        if 'DEPTH' not in df_lwd_full.columns or lwd_log_curve not in df_lwd_full.columns:
            raise KeyError(
                f"Kolom 'DEPTH' atau '{lwd_log_curve}' tidak ditemukan di {os.path.basename(lwd_csv_path)}")

        df_wl = df_wl_full[['DEPTH', ref_log_curve]
                           ].dropna().reset_index(drop=True)
        df_lwd = df_lwd_full[['DEPTH', lwd_log_curve]
                             ].dropna().reset_index(drop=True)

        # 3. Tentukan rentang kedalaman yang sama
        common_min_depth = max(df_wl['DEPTH'].min(), df_lwd['DEPTH'].min())
        common_max_depth = min(df_wl['DEPTH'].max(), df_lwd['DEPTH'].max())

        df_wl_trimmed = df_wl[(df_wl['DEPTH'] >= common_min_depth) & (
            df_wl['DEPTH'] <= common_max_depth)].copy()
        df_lwd_trimmed = df_lwd[(df_lwd['DEPTH'] >= common_min_depth) & (
            df_lwd['DEPTH'] <= common_max_depth)].copy()

        # 4. Ekstrak data log sebagai numpy array
        a1 = df_wl_trimmed[ref_log_curve].values
        a2 = df_lwd_trimmed[lwd_log_curve].values

        minlen = min(len(a1), len(a2))
        a1 = a1[:minlen]
        a2 = a2[:minlen]
        depth_index = df_wl_trimmed['DEPTH'].iloc[:minlen]

        # 5. Jalankan Algoritma COW
        aligner = cow.COW(a2.tolist(), a1.tolist(), nbFrames, slack)
        aligned_a2_raw = aligner.warp_sample_to_target()

        # 6. Siapkan DataFrame hasil
        final_len = min(len(aligned_a2_raw), len(a1))
        output_curve_name = f"{lwd_log_curve}_MATCHED"

        final_df = pd.DataFrame({
            'DEPTH': depth_index.iloc[:final_len],
            ref_log_curve: a1[:final_len],
            lwd_log_curve: a2[:final_len],
            output_curve_name: aligned_a2_raw[:final_len]
        })

        return final_df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File CSV tidak ditemukan: {e.filename}")
    except KeyError as e:
        raise KeyError(
            f"Error pada kolom: {e}. Pastikan nama kolom 'DEPTH' dan kurva log sudah benar.")
    except Exception as e:
        raise RuntimeError(f"Terjx  adi kesalahan saat proses COW: {e}")


def create_before_after_plot_and_summary(df):
    """
    Membaca DataFrame dari MATCHING.csv dan membuat plot perbandingan
    "Before" dan "After" beserta data ringkasan.
    """
    if 'DEPTH' not in df.columns or len(df.columns) < 4:
        raise ValueError(
            "MATCHING.csv harus berisi 'DEPTH' dan setidaknya 3 kurva data.")

    depth_col, ref_col, lwd_col, dm_col = df.columns[0], df.columns[1], df.columns[2], df.columns[3]

    clean_df = df[[ref_col, lwd_col, dm_col]].dropna()
    corr_before, corr_after = np.nan, np.nan
    if len(clean_df) >= 2:
        corr_before = np.corrcoef(clean_df[ref_col], clean_df[lwd_col])[0, 1]
        corr_after = np.corrcoef(clean_df[ref_col], clean_df[dm_col])[0, 1]

    summary_data = {
        "Data Points Used": len(clean_df),
        "Correlation Before": f"{corr_before:.4f}",
        "Correlation After": f"{corr_after:.4f}",
        "Improvement Delta": f"{(corr_after - corr_before):.4f}"
    }

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Before Alignment", "After Alignment"), shared_yaxes=True)
    fig.add_trace(go.Scatter(x=df[ref_col], y=df[depth_col],
                  name=f'Ref ({ref_col})', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[lwd_col], y=df[depth_col], name=f'Original ({lwd_col})', line=dict(
        color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[ref_col], y=df[depth_col], name=f'Ref ({ref_col})', line=dict(
        color='black'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df[dm_col], y=df[depth_col],
                  name=f'Aligned ({dm_col})', line=dict(color='blue')), row=1, col=2)
    fig.update_layout(title_text="Depth Matching Analysis", height=3200, legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(autorange="reversed", title_text="DEPTH")
    fig.update_xaxes(title_text="Curve Value")
    return fig, summary_data
