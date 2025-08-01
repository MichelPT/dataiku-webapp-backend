import numpy as np
import pandas as pd
from services.plotting_service import (
    main_plot,
)
from services.iqual import calculate_iqual


def calculate_interval_statistics(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung statistik (RGBE, RPBE, R-squared) untuk setiap interval contiguous
    di mana IQUAL > 0. Menggunakan metode single-pass untuk efisiensi.
    """
    df = df_input.copy()

    # Inisialisasi kolom hasil dengan NaN
    for col in ['NOD', 'RGBE', 'R_RGBE', 'RPBE', 'R_RPBE']:
        df[col] = np.nan

    # Variabel untuk melacak grup/interval saat ini
    grpstr_idx = None  # Indeks awal dari grup
    grpsize = 0        # Ukuran grup (jumlah titik data)

    # Variabel akumulasi untuk regresi GR vs RT (rg)
    sx_rg, sy_rg, sxy_rg, sx2_rg, sy2_rg = 0.0, 0.0, 0.0, 0.0, 0.0
    # Variabel akumulasi untuk regresi PHIE vs RT (rp)
    sx_rp, sy_rp, sxy_rp, sx2_rp, sy2_rp = 0.0, 0.0, 0.0, 0.0, 0.0

    # Iterasi melalui setiap baris DataFrame
    for idx in range(len(df)):
        iqual = df.at[idx, 'IQUAL']
        iqual_prev = df.at[idx - 1, 'IQUAL'] if idx > 0 else 0

        # Jika kita berada dalam interval yang valid (IQUAL > 0)
        if iqual > 0:
            # Jika ini adalah awal dari interval baru
            if iqual_prev == 0:
                grpstr_idx = idx
                grpsize = 0
                # Reset semua akumulator
                sx_rg, sy_rg, sxy_rg, sx2_rg, sy2_rg = 0.0, 0.0, 0.0, 0.0, 0.0
                sx_rp, sy_rp, sxy_rp, sx2_rp, sy2_rp = 0.0, 0.0, 0.0, 0.0, 0.0

            # Akumulasi nilai untuk perhitungan statistik
            grpsize += 1
            gr = df.at[idx, 'GR']
            rt = df.at[idx, 'RT']
            phie = df.at[idx, 'PHIE']

            # Akumulasi untuk GR vs RT
            sx_rg += gr
            sy_rg += rt
            sxy_rg += gr * rt
            sx2_rg += gr**2
            sy2_rg += rt**2
            # Akumulasi untuk PHIE vs RT
            sx_rp += phie
            sy_rp += rt
            sxy_rp += phie * rt
            sx2_rp += phie**2
            sy2_rp += rt**2

        # Jika kita baru saja keluar dari interval yang valid
        else:
            if grpsize > 0 and grpstr_idx is not None:
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
                for k in range(grpstr_idx, idx):
                    df.at[k, 'NOD'] = grpsize
                    if grpsize > 2:  # Hanya hitung jika data cukup
                        df.at[k, 'RGBE'] = rgbe
                        df.at[k, 'R_RGBE'] = r_rgbe
                        df.at[k, 'RPBE'] = rpbe
                        df.at[k, 'R_RPBE'] = r_rpbe

                # Reset grup
                grpstr_idx = None
                grpsize = 0

    # Proses grup terakhir jika file diakhiri dengan IQUAL > 0
    if grpsize > 0 and grpstr_idx is not None:
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

        for k in range(grpstr_idx, len(df)):
            df.at[k, 'NOD'] = grpsize
            if grpsize > 2:
                df.at[k, 'RGBE'] = rgbe
                df.at[k, 'R_RGBE'] = r_rgbe
                df.at[k, 'RPBE'] = rpbe
                df.at[k, 'R_RPBE'] = r_rpbe

    return df


def process_rgbe_rpbe(df: pd.DataFrame, params=None) -> pd.DataFrame:
    """
    Fungsi utama untuk memproses analisis RGBE-RPBE untuk semua sumur.
    Menggantikan implementasi lama dengan metode yang lebih efisien.
    """
    try:
        # 1. Pastikan kolom IQUAL ada
        df_with_iqual = calculate_iqual(df)

        # 2. Jika tidak ada kolom WELL_NAME, proses seluruh DataFrame sebagai satu sumur
        if 'WELL_NAME' not in df_with_iqual.columns:
            print(
                "Peringatan: Kolom 'WELL_NAME' tidak ditemukan. Memproses seluruh data sebagai satu sumur.")
            df_sorted = df_with_iqual.sort_values(
                by='DEPTH').reset_index(drop=True)
            return calculate_interval_statistics(df_sorted)

        # 3. Proses setiap sumur secara terpisah
        all_results = []
        for well_name, well_df in df_with_iqual.groupby('WELL_NAME'):
            print(f"Memproses sumur: {well_name}")
            # Urutkan data berdasarkan kedalaman untuk setiap sumur
            well_df_sorted = well_df.sort_values(
                by='DEPTH').reset_index(drop=True)
            # Lakukan perhitungan statistik interval
            result_df = calculate_interval_statistics(well_df_sorted)
            all_results.append(result_df)

        # 4. Gabungkan hasil dari semua sumur
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    except Exception as e:
        print(f"Error dalam process_rgbe_rpbe: {str(e)}")
        # Mengembalikan DataFrame asli jika terjadi error untuk mencegah crash
        raise e


def plot_rgbe_rpbe(df):
    """
    Create RGBE-RPBE visualization plot
    """
    sequence_rgbe = ['MARKER', 'GR', 'RT', 'NPHI_RHOB', 'VSH', 'PHIE',
                     'IQUAL', 'RGBE_TEXT', 'RGBE', 'RPBE_TEXT', 'RPBE']
    fig = main_plot(df, sequence_rgbe, title="RGBE Selected Well")

    return fig
