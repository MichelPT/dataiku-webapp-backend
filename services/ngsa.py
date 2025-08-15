import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def _interpolate_coeffs(depth, coeff_df):
    """
    (Fungsi internal ini tidak berubah) Melakukan interpolasi linier pada koefisien regresi.
    """
    if coeff_df.empty:
        return np.array([np.nan] * 4)
    target_cols = ['b0', 'b1', 'b2', 'b3']
    if depth <= coeff_df['DEPTH'].min():
        return coeff_df.iloc[0][target_cols].values
    if depth >= coeff_df['DEPTH'].max():
        return coeff_df.iloc[-1][target_cols].values

    # Menggunakan np.interp untuk efisiensi, yang setara dengan loop interpolasi
    b0 = np.interp(depth, coeff_df['DEPTH'], coeff_df['b0'])
    b1 = np.interp(depth, coeff_df['DEPTH'], coeff_df['b1'])
    b2 = np.interp(depth, coeff_df['DEPTH'], coeff_df['b2'])
    b3 = np.interp(depth, coeff_df['DEPTH'], coeff_df['b3'])
    return np.array([b0, b1, b2, b3])


def process_ngsa_for_well(df_well: pd.DataFrame, params: dict, target_intervals: list, target_zones: list) -> pd.DataFrame:
    """
    Memproses NGSA sesuai dengan alur kerja Loglan (filter GR dinamis).
    """
    gr_col, nphi_col = params.get('GR', 'GR'), params.get('NEUT', 'NPHI')
    required_cols = ['DEPTH', gr_col, nphi_col]
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom hilang: {required_cols}")
        return df_well

    # Filter awal berdasarkan interval/zona dari frontend
    mask = pd.Series(True, index=df_well.index)
    if target_intervals and 'MARKER' in df_well.columns:
        mask &= df_well['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_well.columns:
        mask &= df_well['ZONE'].isin(target_zones)

    df_ngsa = df_well.loc[mask, required_cols].dropna().copy()
    if len(df_ngsa) < 100:
        print(
            f"Peringatan: Data tidak cukup untuk regresi NGSA (hanya {len(df_ngsa)} baris).")
        return df_well

    # --- LANGKAH 1: (SESUAI LOGLAN) HITUNG GR_MAX DINAMIS ---
    print("Langkah 1: Menghitung batas atas GR (GR_MAX) dinamis...")
    window_size = int(params.get('SLIDING_WINDOW', 100))
    gr_caps = []

    # Filter awal untuk data yang wajar sebelum menghitung persentil
    df_filtered_gr = df_ngsa[(df_ngsa[gr_col] > 5) & (df_ngsa[gr_col] < 180)]

    # Menggunakan step 20 seperti di Loglan
    for start in range(0, len(df_filtered_gr) - window_size, 20):
        window = df_filtered_gr.iloc[start:start+window_size]
        if len(window) > window_size / 2:
            gr_cap_value = window[gr_col].quantile(0.98)  # Persentil ke-98
            # Kedalaman representatif
            gr_dep_value = window['DEPTH'].median()
            gr_caps.append({'GR_DEP': gr_dep_value, 'GR_CAP': gr_cap_value})

    if not gr_caps:
        print("Peringatan: Tidak dapat menghitung GR_MAX dinamis, menggunakan nilai default 180.")
        df_ngsa['GR_MAX'] = 180.0
    else:
        gr_cap_df = pd.DataFrame(gr_caps).drop_duplicates(
            subset=['GR_DEP']).sort_values('GR_DEP').reset_index()
        # Interpolasi nilai GR_MAX untuk setiap titik kedalaman
        df_ngsa['GR_MAX'] = np.interp(
            df_ngsa['DEPTH'], gr_cap_df['GR_DEP'], gr_cap_df['GR_CAP'])

    # --- LANGKAH 2: (SESUAI LOGLAN) HITUNG KOEFISIEN REGRESI DENGAN FILTER DINAMIS ---
    print("Langkah 2: Menghitung koefisien regresi dengan filter dinamis...")
    step, min_points = 20, 30

    nphiwat_min = float(params.get('NPHIWAT_MIN', 0.05))
    nphiwat_max = float(params.get('NPHIWAT_MAX', 0.6))
    nphi_filter = (nphiwat_min, nphiwat_max)  # Filter NPHI tetap sama
    coeffs = []

    for start in range(0, len(df_ngsa) - window_size, step):
        window = df_ngsa.iloc[start:start+window_size]

        # Gunakan GR_MAX yang sudah diinterpolasi untuk menyaring data
        mask_filter = (window[gr_col] > 5) & (window[gr_col] < window['GR_MAX']) & \
                      (window[nphi_col] > nphi_filter[0]) & (
                          window[nphi_col] < nphi_filter[1])

        window_filtered = window[mask_filter]

        if len(window_filtered) < min_points:
            continue

        gr_scaled = 0.01 * window_filtered[gr_col]
        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = window_filtered[nphi_col]

        try:
            model = LinearRegression().fit(X, y)
            coeffs.append({
                'DEPTH': window['DEPTH'].mean(),
                'b0': model.intercept_,
                'b1': model.coef_[0],
                'b2': model.coef_[1],
                'b3': model.coef_[2]
            })
        except Exception as e:
            print(
                f"Peringatan: Regresi gagal pada ~{window['DEPTH'].mean()}: {e}")

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return df_well

    coeff_df = pd.DataFrame(coeffs)

    # --- LANGKAH 3: (SESUAI LOGLAN) HITUNG LOG NGSA FINAL ---
    print("Langkah 3: Menghitung log NGSA final...")
    ngsa_list = []
    for _, row in df_ngsa.iterrows():
        depth, gr = row['DEPTH'], row[gr_col]

        # Terapkan filter akhir sebelum kalkulasi, sesuai Loglan
        if not (gr > 5 and gr < row['GR_MAX']):
            ngsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        ngsa_list.append(b0 + b1*grfix + b2*grfix**2 + b3*grfix**3)

    df_ngsa['NGSA'] = ngsa_list

    # Gabungkan kembali hasil ke DataFrame asli yang lengkap
    if 'NGSA' in df_well.columns:
        df_well = df_well.drop(columns=['NGSA'])

    df_merged = pd.merge(
        df_well, df_ngsa[['DEPTH', 'NGSA']], on='DEPTH', how='left')

    if nphi_col in df_merged and 'NGSA' in df_merged:
        df_merged['GAS_EFFECT_NPHI'] = (
            df_merged[nphi_col] < df_merged['NGSA'])
        df_merged['NPHI_DIFF'] = df_merged['NGSA'] - df_merged[nphi_col]

    return df_merged


def process_all_wells_ngsa(df_well: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    print("Memulai proses NGSA...")
    result_df = process_ngsa_for_well(
        df_well, params, target_intervals, target_zones)
    print("✅ Proses NGSA selesai.")
    return result_df
