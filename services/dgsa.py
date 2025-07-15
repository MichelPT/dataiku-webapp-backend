# Nama file: dgsa_processor.py
# Deskripsi: Skrip mandiri dan fleksibel untuk menjalankan kalkulasi DGSA
# berdasarkan parameter yang diberikan.

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


def _interpolate_coeffs(depth, coeff_df):
    """(Internal) Melakukan interpolasi linear pada koefisien regresi."""
    if coeff_df.empty:
        return np.array([np.nan] * 4)

    target_cols = ['b0', 'b1', 'b2', 'b3']

    if depth <= coeff_df['DEPTH'].min():
        return coeff_df.iloc[0][target_cols].values
    if depth >= coeff_df['DEPTH'].max():
        return coeff_df.iloc[-1][target_cols].values

    lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
    upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]

    if upper['DEPTH'] == lower['DEPTH']:
        return lower[target_cols].values

    weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
    interpolated = lower[target_cols] + weight * \
        (upper[target_cols] - lower[target_cols])

    return interpolated.values


def process_dgsa_for_well(df_well: pd.DataFrame, params: dict,
                          marker_column: str = 'MARKER',
                          selected_intervals: list = None) -> Optional[pd.DataFrame]:
    """
    Memproses DGSA untuk satu DataFrame sumur menggunakan parameter dinamis.
    """
    # --- STEP 1: SIAPKAN DATA & VALIDASI ---
    required_cols = ['DEPTH', 'GR', 'RHOB']
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom {required_cols} tidak lengkap.")
        return None

    # --- Buat mask marker dahulu ---
    if marker_column in df_well.columns:
        marker_mask = df_well[marker_column].isin(selected_intervals)
        df_filtered = df_well[marker_mask].copy()
    else:
        df_filtered = df_well.copy()

    # --- Simpan nilai DGSA lama (jika ada) ---
    dgsa_old = df_filtered['DGSA'].copy() if 'DGSA' in df_filtered.columns else pd.Series([
        np.nan]*len(df_filtered), index=df_filtered.index)

    # --- Ambil kolom yang dibutuhkan dan dropna ---
    df_dgsa = df_filtered[required_cols].dropna().copy()

    if len(df_dgsa) < 100:
        print(
            f"Peringatan: Data terlalu sedikit untuk regresi DGSA (hanya {len(df_dgsa)} baris)")
        return None

    # --- STEP 2: SLIDING WINDOW REGRESI DENGAN PARAMETER DINAMIS ---
    window_size = int(params.get('window_size', 106))
    step = int(params.get('step', 20))
    min_points_in_window = int(params.get('min_points_in_window', 30))
    filters = params.get('filters', {})
    gr_filter = filters.get('GR', (5, 180))
    rhob_filter = filters.get('RHOB', (1.5, 3.0))

    coeffs = []

    for start in range(0, len(df_dgsa) - window_size, step):
        window = df_dgsa.iloc[start:start+window_size]
        gr = window['GR'].values
        dens = window['RHOB'].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            dens > rhob_filter[0]) & (dens < rhob_filter[1])
        gr_filtered = gr[mask]
        dens_filtered = dens[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered

        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = dens_filtered

        try:
            model = LinearRegression().fit(X, y)
            if hasattr(model, 'coef_') and len(model.coef_) == 3:
                coeffs.append({
                    'DEPTH': window['DEPTH'].mean(),
                    'b0': model.intercept_, 'b1': model.coef_[0],
                    'b2': model.coef_[1], 'b3': model.coef_[2]
                })
        except Exception as e:
            print(
                f"Peringatan: Gagal melakukan regresi pada kedalaman sekitar {window['DEPTH'].mean()}: {e}")
            continue

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return None

    coeff_df = pd.DataFrame(coeffs)

    # --- STEP 3: INTERPOLASI & HITUNG DGSA ---
    dgsa_list = []
    for _, row in df_dgsa.iterrows():
        depth, gr = row['DEPTH'], row['GR']

        if not (gr_filter[0] < gr < gr_filter[1]):
            dgsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        dgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        dgsa_list.append(dgsa)

    df_dgsa['DGSA'] = dgsa_list

    # --- STEP 4: GABUNGKAN HASIL DAN ANALISIS ---
    df_merged = pd.merge(
        df_well.drop(columns=['DGSA'], errors='ignore'),
        df_dgsa[['DEPTH', 'DGSA']],
        on='DEPTH',
        how='left'
    )

    # Ambil nilai DGSA lama jika ada
    dgsa_old = df_well['DGSA'] if 'DGSA' in df_well.columns else pd.Series(
        [np.nan] * len(df_well), index=df_well.index)

    # Gabungkan DGSA: jika hasil baru NaN, pertahankan nilai lama
    # Gunakan DEPTH sebagai index agar tidak terjadi mismatched assignment
    df_merged.set_index('DEPTH', inplace=True)
    dgsa_old.index = df_well.set_index('DEPTH').index

    # Isi nilai baru hanya jika tidak NaN, selain itu pertahankan nilai lama
    df_merged['DGSA'] = df_merged['DGSA'].combine_first(dgsa_old)

    # Kembalikan index numerik
    df_merged.reset_index(inplace=True)

    if 'RHOB' in df_merged and 'DGSA' in df_merged:
        df_merged['GAS_EFFECT_RHOB'] = (df_merged['RHOB'] < df_merged['DGSA'])
        df_merged['DENS_DIFF'] = df_merged['DGSA'] - df_merged['RHOB']

    return df_merged


def process_all_wells_dgsa(df_well: str, params: dict, selected_intervals: list):
    """
    Fungsi orkestrator utama: memproses DGSA untuk semua sumur dengan parameter kustom.
    """

    unique_wells = df_well['WELL_NAME'].unique()
    print(f"📊 Memproses DGSA untuk {len(unique_wells)} sumur...")

    processed_wells = []
    failed_wells = []

    for i, well_name in enumerate(unique_wells, 1):
        print(f"\n🔍 Memproses sumur {i}/{len(unique_wells)}: {well_name}")
        df_well = df_well[df_well['WELL_NAME'] ==
                          well_name].copy().reset_index(drop=True)

        result_df = process_dgsa_for_well(df_well, params,
                                          marker_column='MARKER',
                                          selected_intervals=selected_intervals)

        if result_df is not None:
            processed_wells.append(result_df)
            print(f"✅ {well_name} - DGSA berhasil dihitung.")
        else:
            failed_wells.append(well_name)
            print(f"❌ {well_name} - DGSA gagal dihitung.")

    if processed_wells:
        df_final = pd.concat(processed_wells, ignore_index=True)

        print(
            f"\n✅ Proses selesai. Hasil gabungan pada file asli.")

        # Menampilkan ringkasan statistik (opsional)
        # print_summary_statistics_dgsa(df_final, len(processed_wells), failed_wells)
    else:
        print("\n❌ Tidak ada sumur yang berhasil diproses!")

    return df_final


def print_summary_statistics_dgsa(df_final, success_count, failed_wells):
    """Mencetak ringkasan statistik dari hasil DGSA."""
    print(f"\n📈 RINGKASAN HASIL DGSA:")
    print(f"  ✅ Sumur berhasil diproses: {success_count}")
    print(f"  ❌ Sumur gagal diproses: {len(failed_wells)}")
    if failed_wells:
        print(f"  ⚠️ Daftar sumur yang gagal: {', '.join(failed_wells)}")

    if 'GAS_EFFECT_RHOB' in df_final:
        total_gas_points = df_final['GAS_EFFECT_RHOB'].sum()
        total_points = len(df_final['GAS_EFFECT_RHOB'].dropna())
        gas_percentage = (total_gas_points / total_points) * \
            100 if total_points > 0 else 0

        print(f"\n🔥 STATISTIK EFEK GAS (BERDASARKAN RHOB):")
        print(f"  Total data points: {total_points:,}")
        print(f"  Points dengan gas effect: {total_gas_points:,}")
        print(f"  Persentase gas effect: {gas_percentage:.2f}%")

        gas_by_well = df_final.groupby('WELL_NAME').agg({
            'GAS_EFFECT_RHOB': ['count', 'sum'],
            'DENS_DIFF': ['mean', 'max']
        }).round(3)
        gas_by_well.columns = ['Total_Points',
                               'Gas_Points', 'Avg_Dens_Diff', 'Max_Dens_Diff']
        gas_by_well['Gas_Percentage'] = (
            gas_by_well['Gas_Points'] / gas_by_well['Total_Points'] * 100).round(2)
        print(f"\n� EFEK GAS PER SUMUR (BERDASARKAN RHOB):")
        print(gas_by_well.to_string())
