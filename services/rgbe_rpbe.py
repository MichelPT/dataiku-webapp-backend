import numpy as np
import pandas as pd
# Asumsi file-file ini ada dan berfungsi
from services.plotting_service import main_plot
from services.iqual import calculate_iqual


def calculate_interval_statistics(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung statistik (RGBE, RPBE, R-squared) untuk setiap interval contiguous
    di mana IQUAL > 0. Menggunakan metode single-pass untuk efisiensi.
    (Fungsi ini tidak perlu diubah, logikanya sudah benar untuk data yang masuk)
    """
    df = df_input.copy()
    if df.empty:
        return df

    # Inisialisasi kolom hasil
    stat_cols = ['NOD', 'RGBE', 'R_RGBE', 'RPBE', 'R_RPBE']
    for col in stat_cols:
        df[col] = np.nan

    # Pastikan kolom input ada
    required_cols = ['IQUAL', 'GR', 'RT', 'PHIE']
    if not all(col in df.columns for col in required_cols):
        print("Peringatan: Kolom yang dibutuhkan (IQUAL, GR, RT, PHIE) tidak ada. Melewatkan kalkulasi statistik.")
        return df_input  # Kembalikan yang asli

    grpstr_idx, grpsize = None, 0
    sx_rg, sy_rg, sxy_rg, sx2_rg, sy2_rg = 0.0, 0.0, 0.0, 0.0, 0.0
    sx_rp, sy_rp, sxy_rp, sx2_rp, sy2_rp = 0.0, 0.0, 0.0, 0.0, 0.0

    # Iterasi melalui setiap baris DataFrame
    for idx in range(len(df)):
        iqual = df.at[idx, 'IQUAL']
        iqual_prev = df.at[idx - 1, 'IQUAL'] if idx > 0 else 0

        if iqual > 0:
            if iqual_prev == 0:
                grpstr_idx = idx
                grpsize = 0
                sx_rg, sy_rg, sxy_rg, sx2_rg, sy2_rg = 0.0, 0.0, 0.0, 0.0, 0.0
                sx_rp, sy_rp, sxy_rp, sx2_rp, sy2_rp = 0.0, 0.0, 0.0, 0.0, 0.0

            grpsize += 1
            gr, rt, phie = df.at[idx, 'GR'], df.at[idx,
                                                   'RT'], df.at[idx, 'PHIE']
            sx_rg += gr
            sy_rg += rt
            sxy_rg += gr * rt
            sx2_rg += gr**2
            sy2_rg += rt**2
            sx_rp += phie
            sy_rp += rt
            sxy_rp += phie * rt
            sx2_rp += phie**2
            sy2_rp += rt**2
        else:
            if grpsize > 2 and grpstr_idx is not None:
                # Hitung statistik untuk grup yang baru saja selesai
                denom_rg = sx_rg * sx_rg - grpsize * sx2_rg
                denom_rp = sx_rp * sx_rp - grpsize * sx2_rp
                denom_r_rg_sq = (grpsize * sx2_rg - sx_rg**2) * \
                    (grpsize * sy2_rg - sy_rg**2)
                denom_r_rp_sq = (grpsize * sx2_rp - sx_rp**2) * \
                    (grpsize * sy2_rp - sy_rp**2)

                rgbe = 100 * (sx_rg * sy_rg - grpsize * sxy_rg) / \
                    denom_rg if denom_rg != 0 else np.nan
                rpbe = (sx_rp * sy_rp - grpsize * sxy_rp) / \
                    denom_rp if denom_rp != 0 else np.nan
                r_rgbe = abs(grpsize * sxy_rg - sx_rg * sy_rg) / \
                    np.sqrt(denom_r_rg_sq) if denom_r_rg_sq > 0 else np.nan
                r_rpbe = abs(grpsize * sxy_rp - sx_rp * sy_rp) / \
                    np.sqrt(denom_r_rp_sq) if denom_r_rp_sq > 0 else np.nan

                # Terapkan hasil ke semua baris dalam grup yang telah selesai
                df.loc[grpstr_idx:idx-1, 'NOD'] = grpsize
                df.loc[grpstr_idx:idx-1, ['RGBE', 'R_RGBE', 'RPBE',
                                          'R_RPBE']] = [rgbe, r_rgbe, rpbe, r_rpbe]

            grpstr_idx, grpsize = None, 0

    # Proses grup terakhir jika file diakhiri dengan IQUAL > 0
    if grpsize > 2 and grpstr_idx is not None:
        denom_rg = sx_rg * sx_rg - grpsize * sx2_rg
        denom_rp = sx_rp * sx_rp - grpsize * sx2_rp
        denom_r_rg_sq = (grpsize * sx2_rg - sx_rg**2) * \
            (grpsize * sy2_rg - sy_rg**2)
        denom_r_rp_sq = (grpsize * sx2_rp - sx_rp**2) * \
            (grpsize * sy2_rp - sy_rp**2)
        rgbe = 100 * (sx_rg * sy_rg - grpsize * sxy_rg) / \
            denom_rg if denom_rg != 0 else np.nan
        rpbe = (sx_rp * sy_rp - grpsize * sxy_rp) / \
            denom_rp if denom_rp != 0 else np.nan
        r_rgbe = abs(grpsize * sxy_rg - sx_rg * sy_rg) / \
            np.sqrt(denom_r_rg_sq) if denom_r_rg_sq > 0 else np.nan
        r_rpbe = abs(grpsize * sxy_rp - sx_rp * sy_rp) / \
            np.sqrt(denom_r_rp_sq) if denom_r_rp_sq > 0 else np.nan
        df.loc[grpstr_idx:, 'NOD'] = grpsize
        df.loc[grpstr_idx:, ['RGBE', 'R_RGBE', 'RPBE', 'R_RPBE']] = [
            rgbe, r_rgbe, rpbe, r_rpbe]

    return df


def process_rgbe_rpbe(df: pd.DataFrame, params: dict = None, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Fungsi utama untuk memproses analisis RGBE-RPBE.
    Sekarang menangani filter interval/zona secara internal untuk mencegah kehilangan data.
    """
    try:
        # 1. Pastikan kolom IQUAL ada di DataFrame lengkap
        df_with_iqual = calculate_iqual(df)

        # 2. Tentukan data mana yang akan diproses
        df_to_process = df_with_iqual.copy()
        has_filters = (target_intervals and 'MARKER' in df.columns) or \
                      (target_zones and 'ZONE' in df.columns)

        if has_filters:
            # Buat mask untuk memfilter data yang akan dihitung
            interval_mask = pd.Series(False, index=df.index)
            if target_intervals and 'MARKER' in df.columns:
                interval_mask |= df['MARKER'].isin(target_intervals)
            if target_zones and 'ZONE' in df.columns:
                interval_mask |= df['ZONE'].isin(target_zones)

            # Hanya proses baris yang cocok dengan filter
            df_to_process = df_with_iqual[interval_mask].copy()

        # 3. Lakukan perhitungan statistik pada data yang telah dipilih (difilter atau lengkap)
        # Fungsi groupby akan menangani jika ada beberapa sumur dalam satu file
        all_results = []
        if 'WELL_NAME' not in df_to_process.columns:
            df_to_process['WELL_NAME'] = 'SINGLE_WELL'

        for well_name, well_df in df_to_process.groupby('WELL_NAME'):
            print(f"Memproses statistik untuk sumur: {well_name}")
            well_df_sorted = well_df.sort_values(
                by='DEPTH').reset_index(drop=True)
            result_df = calculate_interval_statistics(well_df_sorted)
            all_results.append(result_df)

        if not all_results:
            print("Peringatan: Tidak ada data yang diproses setelah filtering.")
            return df  # Kembalikan df asli jika tidak ada hasil

        processed_df = pd.concat(all_results, ignore_index=True)

        # 4. Gabungkan hasil kembali ke DataFrame asli yang lengkap
        stat_cols = ['NOD', 'RGBE', 'R_RGBE', 'RPBE', 'R_RPBE']
        # Hapus kolom lama dari df asli untuk menghindari duplikasi saat merge
        df_final = df.drop(columns=stat_cols, errors='ignore')

        # Hanya merge kolom kunci (DEPTH) dan kolom hasil
        if 'DEPTH' in processed_df.columns:
            df_final = pd.merge(
                df_final,
                processed_df[['DEPTH'] + stat_cols],
                on='DEPTH',
                how='left'
            )
        else:
            print(
                "Error: Kolom 'DEPTH' tidak ditemukan di hasil proses, tidak dapat menggabungkan.")
            return df

        return df_final

    except Exception as e:
        print(f"Error dalam process_rgbe_rpbe: {str(e)}")
        raise e


def plot_rgbe_rpbe(df):
    """
    Membuat plot visualisasi RGBE-RPBE.
    """
    sequence_rgbe = ['MARKER', 'GR', 'RT', 'NPHI_RHOB', 'VSH', 'PHIE',
                     'IQUAL', 'RGBE_TEXT', 'RGBE', 'RPBE_TEXT', 'RPBE']
    fig = main_plot(df, sequence_rgbe, title="RGBE Selected Well")
    return fig
