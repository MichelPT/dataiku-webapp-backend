# Nama file: rgsa_processor.py
# Deskripsi: Skrip mandiri dan fleksibel untuk menjalankan kalkulasi RGSA
# berdasarkan parameter yang diberikan.

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# FIX: Fungsi inti sekarang menerima dictionary 'params' untuk kustomisasi


def process_rgsa_for_well(df_well: pd.DataFrame, params: dict,
                          marker_column: str = 'MARKER',
                          selected_intervals: list = None) -> Optional[pd.DataFrame]:
    """
    Memproses RGSA untuk satu DataFrame sumur menggunakan parameter dinamis.
    """
    # --- STEP 1: SIAPKAN DATA & VALIDASI ---
    required_cols = ['DEPTH', 'GR', 'RT']
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom {required_cols} tidak lengkap.")
        return None

    print(selected_intervals)

    # --- Buat mask marker dahulu ---
    if marker_column in df_well.columns:
        marker_mask = df_well[marker_column].isin(selected_intervals)
        df_filtered = df_well[marker_mask].copy()
    else:
        df_filtered = df_well.copy()

    # --- Simpan nilai RGSA lama (jika ada) ---
    rgsa_old = df_filtered['RGSA'].copy() if 'RGSA' in df_filtered.columns else pd.Series([
        np.nan]*len(df_filtered), index=df_filtered.index)

    # --- Ambil kolom yang dibutuhkan dan dropna ---
    df_rgsa = df_filtered[required_cols].dropna().copy()

    if len(df_rgsa) < 100:
        print(
            f"Peringatan: Data terlalu sedikit untuk regresi RGSA (hanya {len(df_rgsa)} baris)")
        return None

    # --- STEP 2: SLIDING WINDOW REGRESI DENGAN PARAMETER DINAMIS ---
    # Ambil nilai dari dictionary params, gunakan default jika tidak ada
    window_size = int(params.get('window_size', 106))
    step = int(params.get('step', 20))
    min_points_in_window = int(params.get('min_points_in_window', 30))
    # Ambil filter dari dictionary params
    filters = params.get('filters', {})
    gr_filter = filters.get('GR', (5, 180))
    rt_filter = filters.get('RT', (0.1, 1000))

    coeffs = []

    for start in range(0, len(df_rgsa) - window_size, step):
        window = df_rgsa.iloc[start:start+window_size]
        gr = window['GR'].values
        rt = window['RT'].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            rt > rt_filter[0]) & (rt < rt_filter[1])
        gr_filtered = gr[mask]
        rt_filtered = rt[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered
        log_rt = np.log10(rt_filtered)

        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = log_rt

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

    # --- STEP 3: INTERPOLASI & HITUNG RGSA ---
    def interpolate_coeffs(depth):
        if depth <= coeff_df['DEPTH'].min():
            return coeff_df.iloc[0]
        if depth >= coeff_df['DEPTH'].max():
            return coeff_df.iloc[-1]
        lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
        upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]
        if upper['DEPTH'] == lower['DEPTH']:
            return lower
        weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
        return lower + weight * (upper - lower)

    rgsa_list = []
    for _, row in df_rgsa.iterrows():
        depth, gr = row['DEPTH'], row['GR']

        if not (gr_filter[0] < gr < gr_filter[1]):
            rgsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = interpolate_coeffs(
            depth)[['b0', 'b1', 'b2', 'b3']].values
        grfix = 0.01 * gr
        log_rgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        rgsa = 10**log_rgsa
        rgsa_list.append(rgsa)

    df_rgsa['RGSA'] = rgsa_list

    # --- STEP 4: GABUNGKAN HASIL DAN ANALISIS ---
    # Merge hasil RGSA baru ke df_well
    df_merged = pd.merge(
        df_well.drop(columns=['RGSA'], errors='ignore'),
        df_rgsa[['DEPTH', 'RGSA']],
        on='DEPTH',
        how='left'
    )

    # Ambil nilai RGSA lama jika ada
    rgsa_old = df_well['RGSA'] if 'RGSA' in df_well.columns else pd.Series(
        [np.nan] * len(df_well), index=df_well.index)

    # Gabungkan RGSA: jika hasil baru NaN, pertahankan nilai lama
    # Gunakan DEPTH sebagai index agar tidak terjadi mismatched assignment
    df_merged.set_index('DEPTH', inplace=True)
    rgsa_old.index = df_well.set_index('DEPTH').index

    # Isi nilai baru hanya jika tidak NaN, selain itu pertahankan nilai lama
    df_merged['RGSA'] = df_merged['RGSA'].combine_first(rgsa_old)

    # Kembalikan index numerik
    df_merged.reset_index(inplace=True)

    if 'RT' in df_merged and 'RGSA' in df_merged:
        df_merged['GAS_EFFECT_RT'] = (df_merged['RT'] > df_merged['RGSA'])
        df_merged['RT_RATIO'] = df_merged['RT'] / df_merged['RGSA']
        df_merged['RT_DIFF'] = df_merged['RT'] - df_merged['RGSA']

    return df_merged


def process_all_wells_rgsa(df_well: pd.DataFrame, params: dict, selected_intervals: list):
    """
    Fungsi orkestrator utama: memproses RGSA untuk semua sumur dengan parameter kustom.
    """

    unique_wells = df_well['WELL_NAME'].unique()
    print(f"📊 Memproses RGSA untuk {len(unique_wells)} sumur...")

    processed_wells = []
    failed_wells = []

    for i, well_name in enumerate(unique_wells, 1):
        print(f"\n🔍 Memproses sumur {i}/{len(unique_wells)}: {well_name}")
        df_well = df_well[df_well['WELL_NAME'] ==
                          well_name].copy().reset_index(drop=True)

        # Teruskan dictionary `params` ke fungsi pemrosesan
        result_df = process_rgsa_for_well(df_well, params,
                                          marker_column='MARKER',
                                          selected_intervals=selected_intervals)

        if result_df is not None:
            processed_wells.append(result_df)
            print(f"✅ {well_name} - RGSA berhasil dihitung.")
        else:
            failed_wells.append(well_name)
            print(f"❌ {well_name} - RGSA gagal dihitung.")

    if processed_wells:
        df_final = pd.concat(processed_wells, ignore_index=True)
        print(
            f"\n✅ Proses selesai. Hasil gabungan pada file asli.")

        # print_summary_statistics(df_final, len(processed_wells), failed_wells)
    else:
        print("\n❌ Tidak ada sumur yang berhasil diproses!")

    return df_final


def print_summary_statistics(df_final, success_count, failed_wells):
    """Mencetak ringkasan statistik dari hasil RGSA."""
    print(f"\n📈 RINGKASAN HASIL RGSA:")
    print(f"  ✅ Sumur berhasil diproses: {success_count}")
    print(f"  ❌ Sumur gagal diproses: {len(failed_wells)}")
    if failed_wells:
        print(f"  ⚠️ Daftar sumur yang gagal: {', '.join(failed_wells)}")

    if 'GAS_EFFECT_RT' in df_final:
        total_gas_points = df_final['GAS_EFFECT_RT'].sum()
        total_points = len(df_final['GAS_EFFECT_RT'].dropna())
        gas_percentage = (total_gas_points / total_points) * \
            100 if total_points > 0 else 0

        print(f"\n🔥 STATISTIK EFEK GAS (BERDASARKAN RT):")
        print(f"  Total data points: {total_points:,}")
        print(f"  Points dengan gas effect: {total_gas_points:,}")
        print(f"  Persentase gas effect: {gas_percentage:.2f}%")

        gas_by_well = df_final.groupby('WELL_NAME').agg({
            'GAS_EFFECT_RT': ['count', 'sum'],
            'RT_RATIO': ['mean', 'max'],
            'RT_DIFF': ['mean', 'max']
        }).round(3)
        gas_by_well.columns = ['Total_Points', 'Gas_Points',
                               'Avg_RT_Ratio', 'Max_RT_Ratio', 'Avg_RT_Diff', 'Max_RT_Diff']
        gas_by_well['Gas_Percentage'] = (
            gas_by_well['Gas_Points'] / gas_by_well['Total_Points'] * 100).round(2)
        print(f"\n📋 EFEK GAS PER SUMUR (BERDASARKAN RT):")
        print(gas_by_well.to_string())
