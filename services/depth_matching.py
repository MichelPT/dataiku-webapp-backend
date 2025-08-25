import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import os


def import_cow():
    """
    Mengimpor library COW dari path yang spesifik.
    Pastikan path ini benar di lingkungan server Anda.
    """
    cow_path = Path(
        "D:/DATAIKU/PROJECT PERTAMINA/dataiku_webapp/api/utils/cow_fixed.py")

    if not cow_path.exists():
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

        df_wl_full.rename(columns={'DEPT': 'DEPTH'}, inplace=True)
        df_lwd_full.rename(columns={'DEPT': 'DEPTH'}, inplace=True)

        # 2. Validasi dan siapkan DataFrame, bersihkan data NaN
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

        # 3. Tentukan dan terapkan rentang kedalaman yang sama (overlapping)
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
        a1_list = a1.tolist()
        a2_list = a2.tolist()

        aligner = cow.COW(a2_list, a1_list, nbFrames, slack)
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
            f"Error pada kolom: {e}. Pastikan nama kolom 'DEPTH' dan nama kurva log sudah benar di file CSV.")
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat proses COW: {e}")
