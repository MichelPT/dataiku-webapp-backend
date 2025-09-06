# import sys
# import os
# import importlib.util
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# def import_cow():
#     """
#     Mengimpor library COW dari path yang spesifik.
#     Pastikan path ini benar di lingkungan server Anda.
#     """
#     cow_path = "utils/cow_fixed.py"

#     if not os.path.exists(cow_path):
#         raise ImportError(
#             f"Tidak dapat menemukan file 'cow_fixed.py' di: {cow_path}")

#     try:
#         spec = importlib.util.spec_from_file_location(
#             "cow_fixed", str(cow_path))
#         mod = importlib.util.module_from_spec(spec)
#         sys.modules["cow_fixed"] = mod
#         spec.loader.exec_module(mod)
#         return mod
#     except Exception as e:
#         raise ImportError(f"Gagal mengimpor 'cow_fixed.py': {e}")


# cow = import_cow()


# def run_depth_matching(ref_csv_path: str, lwd_csv_path: str, ref_log_curve: str, lwd_log_curve: str, nbFrames: int, slack: int):
#     """
#     Fungsi utama untuk menjalankan proses depth matching dari file CSV.
#     """
#     try:
#         # 1. Baca file CSV
#         df_wl_full = pd.read_csv(ref_csv_path)
#         df_lwd_full = pd.read_csv(lwd_csv_path)

#         # Ganti nama 'DEPT' ke 'DEPTH' jika ada, tanpa mengubah DataFrame asli di memori
#         if 'DEPT' in df_wl_full.columns:
#             df_wl_full = df_wl_full.rename(columns={'DEPT': 'DEPTH'})
#         if 'DEPT' in df_lwd_full.columns:
#             df_lwd_full = df_lwd_full.rename(columns={'DEPT': 'DEPTH'})

#         # 2. Validasi dan siapkan DataFrame
#         if 'DEPTH' not in df_wl_full.columns or ref_log_curve not in df_wl_full.columns:
#             raise KeyError(
#                 f"Kolom 'DEPTH' atau '{ref_log_curve}' tidak ditemukan di {os.path.basename(ref_csv_path)}")
#         if 'DEPTH' not in df_lwd_full.columns or lwd_log_curve not in df_lwd_full.columns:
#             raise KeyError(
#                 f"Kolom 'DEPTH' atau '{lwd_log_curve}' tidak ditemukan di {os.path.basename(lwd_csv_path)}")

#         df_wl = df_wl_full[['DEPTH', ref_log_curve]
#                            ].dropna().reset_index(drop=True)
#         df_lwd = df_lwd_full[['DEPTH', lwd_log_curve]
#                              ].dropna().reset_index(drop=True)

#         # 3. Tentukan rentang kedalaman yang sama
#         common_min_depth = max(df_wl['DEPTH'].min(), df_lwd['DEPTH'].min())
#         common_max_depth = min(df_wl['DEPTH'].max(), df_lwd['DEPTH'].max())

#         df_wl_trimmed = df_wl[(df_wl['DEPTH'] >= common_min_depth) & (
#             df_wl['DEPTH'] <= common_max_depth)].copy()
#         df_lwd_trimmed = df_lwd[(df_lwd['DEPTH'] >= common_min_depth) & (
#             df_lwd['DEPTH'] <= common_max_depth)].copy()

#         # 4. Ekstrak data log sebagai numpy array
#         a1 = df_wl_trimmed[ref_log_curve].values
#         a2 = df_lwd_trimmed[lwd_log_curve].values

#         minlen = min(len(a1), len(a2))
#         a1 = a1[:minlen]
#         a2 = a2[:minlen]
#         depth_index = df_wl_trimmed['DEPTH'].iloc[:minlen]

#         # 5. Jalankan Algoritma COW
#         aligner = cow.COW(a2.tolist(), a1.tolist(), nbFrames, slack)
#         aligned_a2_raw = aligner.warp_sample_to_target()

#         # 6. Siapkan DataFrame hasil
#         final_len = min(len(aligned_a2_raw), len(a1))
#         output_curve_name = f"{lwd_log_curve}_MATCHED"

#         final_df = pd.DataFrame({
#             'DEPTH': depth_index.iloc[:final_len],
#             ref_log_curve: a1[:final_len],
#             lwd_log_curve: a2[:final_len],
#             output_curve_name: aligned_a2_raw[:final_len]
#         })

#         return final_df

#     except FileNotFoundError as e:
#         raise FileNotFoundError(f"File CSV tidak ditemukan: {e.filename}")
#     except KeyError as e:
#         raise KeyError(
#             f"Error pada kolom: {e}. Pastikan nama kolom 'DEPTH' dan kurva log sudah benar.")
#     except Exception as e:
#         raise RuntimeError(f"Terjx  adi kesalahan saat proses COW: {e}")


# def create_before_after_plot_and_summary(df):
#     """
#     Membaca DataFrame dari MATCHING.csv dan membuat plot perbandingan
#     "Before" dan "After" beserta data ringkasan.
#     """
#     if 'DEPTH' not in df.columns or len(df.columns) < 4:
#         raise ValueError(
#             "MATCHING.csv harus berisi 'DEPTH' dan setidaknya 3 kurva data.")

#     depth_col, ref_col, lwd_col, dm_col = df.columns[0], df.columns[1], df.columns[2], df.columns[3]

#     clean_df = df[[ref_col, lwd_col, dm_col]].dropna()
#     corr_before, corr_after = np.nan, np.nan
#     if len(clean_df) >= 2:
#         corr_before = np.corrcoef(clean_df[ref_col], clean_df[lwd_col])[0, 1]
#         corr_after = np.corrcoef(clean_df[ref_col], clean_df[dm_col])[0, 1]

#     summary_data = {
#         "Data Points Used": len(clean_df),
#         "Correlation Before": f"{corr_before:.4f}",
#         "Correlation After": f"{corr_after:.4f}",
#         "Improvement Delta": f"{(corr_after - corr_before):.4f}"
#     }

#     fig = make_subplots(rows=1, cols=2, subplot_titles=(
#         "Before Alignment", "After Alignment"), shared_yaxes=True)
#     fig.add_trace(go.Scatter(x=df[ref_col], y=df[depth_col],
#                   name=f'Ref ({ref_col})', line=dict(color='black')), row=1, col=1)
#     fig.add_trace(go.Scatter(x=df[lwd_col], y=df[depth_col], name=f'Original ({lwd_col})', line=dict(
#         color='red', dash='dash')), row=1, col=1)
#     fig.add_trace(go.Scatter(x=df[ref_col], y=df[depth_col], name=f'Ref ({ref_col})', line=dict(
#         color='black'), showlegend=False), row=1, col=2)
#     fig.add_trace(go.Scatter(x=df[dm_col], y=df[depth_col],
#                   name=f'Aligned ({dm_col})', line=dict(color='blue')), row=1, col=2)
#     fig.update_layout(title_text="Depth Matching Analysis", height=3200, legend=dict(
#         orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
#     fig.update_yaxes(autorange="reversed", title_text="DEPTH")
#     fig.update_xaxes(title_text="Curve Value")
#     return fig, summary_data
import pandas as pd
import numpy as np
from dtaidistance import dtw
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import sys
import importlib.util
from pathlib import Path

# --- FUNGSI INTERNAL (HELPER) ---


def import_cow():
    """Mengimpor library COW dari path spesifik."""
    cow_path = Path("utils/cow_fixed.py")
    if not cow_path.exists():
        raise ImportError(
            f"File 'cow_fixed.py' tidak ditemukan di: {cow_path.resolve()}")
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


def preprocess_signal(signal, method='savgol', window=5):
    """
    Preprocessing sinyal untuk mengurangi noise dan meningkatkan kualitas alignment.
    """
    if method == 'savgol':
        # Savitzky-Golay filter untuk smoothing yang mempertahankan pola
        if len(signal) > window:
            return savgol_filter(signal, window_length=window, polyorder=2)
    elif method == 'gaussian':
        # Gaussian filter untuk smoothing yang lebih halus
        return gaussian_filter1d(signal, sigma=1.0)
    return signal


def adaptive_normalize_for_warping(series, method='robust'):
    """
    Normalisasi adaptif yang lebih robust untuk proses warping.
    """
    if method == 'robust':
        # Menggunakan median dan MAD untuk normalisasi yang lebih robust terhadap outliers
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return np.zeros_like(series)
        # 1.4826 adalah faktor untuk konsistensi dengan std
        return (series - median) / (1.4826 * mad)
    elif method == 'minmax':
        # Min-max normalization
        min_val, max_val = np.min(series), np.max(series)
        if max_val == min_val:
            return np.zeros_like(series)
        return (series - min_val) / (max_val - min_val)
    else:  # standard z-score
        std_dev = np.std(series)
        if std_dev == 0:
            return np.zeros_like(series)
        return (series - np.mean(series)) / std_dev


def advanced_dynamic_scaling(source_signal, target_signal, window_size=51, alpha=0.7):
    """
    Scaling dinamis yang lebih advanced dengan kombinasi metode lokal dan global.
    """
    if window_size % 2 == 0:
        window_size += 1

    source_s = pd.Series(source_signal)
    target_s = pd.Series(target_signal)

    # Rolling statistics dengan handling untuk window edges
    source_mean = source_s.rolling(
        window=window_size, center=True, min_periods=max(1, window_size//4)).mean()
    target_mean = target_s.rolling(
        window=window_size, center=True, min_periods=max(1, window_size//4)).mean()
    source_std = source_s.rolling(
        window=window_size, center=True, min_periods=max(1, window_size//4)).std()
    target_std = target_s.rolling(
        window=window_size, center=True, min_periods=max(1, window_size//4)).std()

    # Avoid division by zero
    source_std = source_std.fillna(np.std(source_signal))
    target_std = target_std.fillna(np.std(target_signal))
    source_std = np.where(source_std == 0, np.std(source_signal), source_std)
    target_std = np.where(target_std == 0, np.std(target_signal), target_std)

    # Local scaling
    local_scaled = target_mean + \
        (source_s - source_mean) * (target_std / source_std)

    # Global scaling sebagai baseline
    global_scaled = global_scale_values(source_signal, target_signal)

    # Kombinasi adaptive antara local dan global scaling
    # Alpha mengontrol seberapa banyak pengaruh local vs global
    final_scaled = alpha * local_scaled + (1 - alpha) * global_scaled

    return final_scaled.values


def global_scale_values(source_signal, target_signal):
    """Metode scaling global yang disempurnakan."""
    # Menggunakan robust statistics
    source_median = np.median(source_signal)
    target_median = np.median(target_signal)
    source_mad = np.median(np.abs(source_signal - source_median))
    target_mad = np.median(np.abs(target_signal - target_median))

    if source_mad == 0 or target_mad == 0:
        return source_signal

    # Robust scaling
    return target_median + (source_signal - source_median) * (target_mad / source_mad)


def iterative_dtw_alignment(lwd_signal, ref_signal, lwd_depths, ref_depths, max_iterations=3):
    """
    DTW iteratif untuk meningkatkan kualitas alignment.
    """
    current_lwd = lwd_signal.copy()
    current_depths = lwd_depths.copy()

    for iteration in range(max_iterations):
        # Normalisasi untuk DTW
        norm_ref = adaptive_normalize_for_warping(ref_signal, method='robust')
        norm_lwd = adaptive_normalize_for_warping(current_lwd, method='robust')

        # Jalankan DTW
        path = dtw.warping_path(norm_lwd, norm_ref)

        # Ekstrak mapping dari path
        ref_indices = [p[1] for p in path]
        lwd_indices = [p[0] for p in path]

        # Buat mapping kedalaman
        mapped_ref_depths = ref_depths[ref_indices]
        mapped_lwd_depths = current_depths[lwd_indices]

        # Buat fungsi warping yang halus
        if len(np.unique(mapped_ref_depths)) > 1:
            warping_func = interp1d(
                mapped_ref_depths, mapped_lwd_depths,
                kind='linear', bounds_error=False,
                fill_value='extrapolate'
            )

            # Apply warping
            warped_depths = warping_func(ref_depths)

            # Interpolasi nilai LWD pada kedalaman baru
            if len(np.unique(current_depths)) > 1:
                lwd_interp = interp1d(
                    current_depths, current_lwd,
                    kind='linear', bounds_error=False,
                    fill_value='extrapolate'
                )
                current_lwd = lwd_interp(warped_depths)
                current_depths = ref_depths.copy()

    return current_lwd


def enhanced_cow_alignment(source_signal, target_signal, num_chunks, slack):
    """
    COW alignment yang ditingkatkan dengan preprocessing dan post-processing.
    """
    # Preprocessing: smoothing untuk mengurangi noise
    smooth_source = preprocess_signal(source_signal, method='savgol', window=5)
    smooth_target = preprocess_signal(target_signal, method='savgol', window=5)

    # COW alignment
    aligner = cow.COW(smooth_source.tolist(), smooth_target.tolist(),
                      nbFrames=num_chunks, slack=slack)
    aligned_signal = np.array(aligner.warp_sample_to_target())

    return aligned_signal


def calculate_alignment_quality(aligned_signal, target_signal):
    """
    Menghitung kualitas alignment menggunakan berbagai metrik.
    """
    # Pastikan panjang yang sama
    min_len = min(len(aligned_signal), len(target_signal))
    aligned_signal = aligned_signal[:min_len]
    target_signal = target_signal[:min_len]

    # Correlation coefficient
    correlation = np.corrcoef(aligned_signal, target_signal)[
        0, 1] if min_len > 1 else 0

    # RMSE
    rmse = np.sqrt(np.mean((aligned_signal - target_signal) ** 2))

    # MAE
    mae = np.mean(np.abs(aligned_signal - target_signal))

    return {
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'data_points': min_len
    }


def run_enhanced_depth_matching(ref_path: str, lwd_path: str, ref_curve: str,
                                lwd_curve: str, num_chunks: int = 10, slack: int = 3):
    """
    Enhanced depth matching dengan multiple algoritma dan quality assessment.
    """
    print("=== Enhanced Depth Matching Process ===")

    # 1. Load dan persiapan data
    print("1. Loading data...")
    ref_df_full = pd.read_csv(ref_path)
    lwd_df_full = pd.read_csv(lwd_path)

    # Rename DEPT to DEPTH if necessary
    if 'DEPT' in ref_df_full.columns:
        ref_df_full.rename(columns={'DEPT': 'DEPTH'}, inplace=True)
    if 'DEPT' in lwd_df_full.columns:
        lwd_df_full.rename(columns={'DEPT': 'DEPTH'}, inplace=True)

    # Persiapan data untuk processing
    ref_proc = ref_df_full[['DEPTH', ref_curve]].dropna(
    ).sort_values(by='DEPTH').reset_index(drop=True)
    lwd_proc = lwd_df_full[['DEPTH', lwd_curve]].dropna(
    ).sort_values(by='DEPTH').reset_index(drop=True)

    # Tentukan overlap range
    min_depth = max(ref_proc['DEPTH'].min(), lwd_proc['DEPTH'].min())
    max_depth = min(ref_proc['DEPTH'].max(), lwd_proc['DEPTH'].max())

    # Filter ke overlap range
    ref_proc = ref_proc[(ref_proc['DEPTH'] >= min_depth) &
                        (ref_proc['DEPTH'] <= max_depth)].reset_index(drop=True)
    lwd_proc = lwd_proc[(lwd_proc['DEPTH'] >= min_depth) &
                        (lwd_proc['DEPTH'] <= max_depth)].reset_index(drop=True)

    print(f"   Working range: {min_depth:.2f} - {max_depth:.2f}")
    print(f"   Ref points: {len(ref_proc)}, LWD points: {len(lwd_proc)}")

    # Ekstrak signals
    ref_signal = ref_proc[ref_curve].values
    lwd_signal = lwd_proc[lwd_curve].values
    ref_depths = ref_proc['DEPTH'].values
    lwd_depths = lwd_proc['DEPTH'].values

    # 2. Iterative DTW Alignment
    print("2. Running iterative DTW alignment...")
    dtw_aligned = iterative_dtw_alignment(lwd_signal, ref_signal,
                                          lwd_depths, ref_depths, max_iterations=3)

    dtw_quality = calculate_alignment_quality(dtw_aligned, ref_signal)
    print(f"   DTW Quality - Correlation: {dtw_quality['correlation']:.4f}")

    # 3. Enhanced COW Alignment
    print("3. Running enhanced COW alignment...")
    cow_aligned = enhanced_cow_alignment(
        dtw_aligned, ref_signal, num_chunks, slack)

    cow_quality = calculate_alignment_quality(cow_aligned, ref_signal)
    print(f"   COW Quality - Correlation: {cow_quality['correlation']:.4f}")

    # 4. Advanced Dynamic Value Scaling
    print("4. Applying advanced dynamic scaling...")
    final_aligned = advanced_dynamic_scaling(cow_aligned, ref_signal,
                                             window_size=51, alpha=0.7)

    final_quality = calculate_alignment_quality(final_aligned, ref_signal)
    print(
        f"   Final Quality - Correlation: {final_quality['correlation']:.4f}")

    # 5. Interpolasi ke seluruh range referensi
    print("5. Interpolating to full reference range...")

    # Buat interpolator dari hasil final
    final_interpolator = interp1d(
        ref_depths, final_aligned,
        kind='cubic', bounds_error=False, fill_value=np.nan
    )

    # Interpolator untuk LWD original
    lwd_original_interp = interp1d(
        lwd_df_full['DEPTH'], lwd_df_full[lwd_curve],
        kind='linear', bounds_error=False, fill_value=np.nan
    )

    # Apply ke seluruh range
    full_range_dm = final_interpolator(ref_df_full['DEPTH'])
    full_range_lwd_original = lwd_original_interp(ref_df_full['DEPTH'])

    # 6. Buat hasil DataFrame
    result_df = pd.DataFrame({
        'DEPTH': ref_df_full['DEPTH'],
        ref_curve: ref_df_full[ref_curve],
        lwd_curve: full_range_lwd_original,
        f"{lwd_curve}_DM": full_range_dm
    })

    # Quality assessment untuk hasil akhir
    clean_result = result_df.dropna()
    if len(clean_result) > 1:
        final_corr_original = np.corrcoef(clean_result[ref_curve],
                                          clean_result[lwd_curve])[0, 1]
        final_corr_aligned = np.corrcoef(clean_result[ref_curve],
                                         clean_result[f"{lwd_curve}_DM"])[0, 1]
        improvement = final_corr_aligned - final_corr_original

        print(f"\n=== Final Results ===")
        print(f"Original correlation: {final_corr_original:.4f}")
        print(f"Aligned correlation:  {final_corr_aligned:.4f}")
        print(f"Improvement:         {improvement:.4f}")
        print(f"Data points:         {len(clean_result)}")

    return result_df.dropna(subset=[ref_curve])


# Wrapper function untuk kompatibilitas dengan kode lama
def run_depth_matching(ref_path: str, lwd_path: str, ref_curve: str,
                       lwd_curve: str, num_chunks: int, slack: int):
    """
    Wrapper function yang kompatibel dengan signature asli.
    """
    return run_enhanced_depth_matching(ref_path, lwd_path, ref_curve,
                                       lwd_curve, num_chunks, slack)
