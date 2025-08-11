import numpy as np
import pandas as pd


def calc_gradient(x1, y1, x2, y2):
    with np.errstate(divide='ignore', invalid='ignore'):
        grad = np.abs(y2 - y1) / np.abs(x2 - x1)
        grad = np.where(np.isinf(grad), np.nan, grad)
    return grad


def calc_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)


def calculate_nphi_rhob_intersection(df: pd.DataFrame, prcnt_qz: float, prcnt_wtr: float) -> dict:
    """
    FUNGSI INTI TERPUSAT: Menghitung titik potong NPHI-RHOB.
    Dapat digunakan oleh endpoint manapun.
    """
    df_clean = df[['NPHI', 'RHOB']].dropna()
    if df_clean.empty:
        raise ValueError("Tidak ada data NPHI & RHOB yang valid.")

    x_quartz, y_quartz = -0.02, 2.65
    x_water, y_water = 1.0, 1.0

    df_clean["GRAD_QZ"] = calc_gradient(
        x_quartz, y_quartz, df_clean["NPHI"], df_clean["RHOB"])
    df_QZ = df_clean[df_clean['GRAD_QZ'] > 0].sort_values(by='GRAD_QZ')
    if df_QZ.empty:
        raise ValueError("Tidak bisa menentukan garis QZ.")
    df_QZ = df_QZ.iloc[int(prcnt_qz / 100 * len(df_QZ)):, :]
    if df_QZ.empty:
        raise ValueError("Data QZ kosong setelah filtering percentile.")

    df_clean["GRAD_WTR"] = calc_gradient(
        x_water, y_water, df_clean["RHOB"], df_clean["NPHI"])
    df_WTR = df_clean[df_clean['GRAD_WTR'] > 0].sort_values(by='GRAD_WTR')
    if df_WTR.empty:
        raise ValueError("Tidak bisa menentukan garis WTR.")
    df_WTR = df_WTR.iloc[int(prcnt_wtr / 100 * len(df_WTR)):, :]
    if df_WTR.empty:
        raise ValueError("Data WTR kosong setelah filtering percentile.")

    pq1 = (0.0, y_quartz - y_quartz)
    pq2 = (df_QZ.iloc[0]['NPHI'], (y_quartz - df_QZ.iloc[0]['RHOB']))
    pw1 = (df_WTR.iloc[0]['NPHI'], (y_quartz - df_WTR.iloc[0]['RHOB']))
    pw2 = (x_water, y_quartz - y_water)

    intersect = calc_intersection(pq1, pq2, pw1, pw2)

    if intersect:
        xx0, yy0 = intersect[0], y_quartz - intersect[1]
        return {"nphi_sh": round(xx0, 4), "rhob_sh": round(yy0, 4)}
    else:
        raise ValueError("Tidak ditemukan titik potong (intersection).")


def find_closest_point_value(df: pd.DataFrame, x_col: str, y_col: str, target_x: float, target_y: float, value_col: str):
    """
    Fungsi helper untuk mencari nilai di 'value_col' dari baris data
    yang paling dekat dengan titik target (target_x, target_y).
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns or value_col not in df.columns:
        return None

    # Pastikan data bersih dari NaN pada kolom yang akan digunakan
    df_clean = df[[x_col, y_col, value_col]].dropna()
    if df_clean.empty:
        return None

    # Hitung jarak kuadrat Euclidean (lebih cepat dari akar kuadrat)
    distances = (df_clean[x_col] - target_x)**2 + \
        (df_clean[y_col] - target_y)**2

    # Dapatkan indeks dari baris dengan jarak terkecil
    closest_idx = distances.idxmin()

    # Kembalikan nilai dari kolom yang diinginkan pada baris tersebut
    closest_value = df_clean.loc[closest_idx, value_col]

    return closest_value


def calculate_gr_ma_sh_from_nphi_rhob(df: pd.DataFrame, prcnt_qz: float, prcnt_wtr: float) -> dict:
    """
    Menghitung gr_ma dan gr_sh berdasarkan titik terdekat pada crossplot NPHI vs RHOB.

    Returns:
        dict: Berisi {'gr_ma': float, 'gr_sh': float}
    """
    # 1. Pastikan kolom yang dibutuhkan ada
    required_cols = ['NPHI', 'RHOB', 'GR']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom yang dibutuhkan '{col}' tidak ditemukan.")

    # 2. Hitung titik shale (intersection) terlebih dahulu
    try:
        shale_point = calculate_nphi_rhob_intersection(df, prcnt_qz, prcnt_wtr)
        nphi_sh = shale_point['nphi_sh']
        rhob_sh = shale_point['rhob_sh']
    except ValueError as e:
        # Jika intersection gagal, kita tidak bisa menghitung gr_sh
        raise ValueError(f"Gagal menghitung titik shale: {e}")

    # 3. Definisikan titik referensi
    x_quartz, y_quartz = -0.02, 2.65

    # 4. Cari nilai GR terdekat dari titik Quartz untuk mendapatkan gr_ma
    gr_ma = find_closest_point_value(
        df=df,
        x_col='NPHI',
        y_col='RHOB',
        target_x=x_quartz,
        target_y=y_quartz,
        value_col='GR'
    )

    # 5. Cari nilai GR terdekat dari titik Shale (intersection) untuk mendapatkan gr_sh
    gr_sh = find_closest_point_value(
        df=df,
        x_col='NPHI',
        y_col='RHOB',
        target_x=nphi_sh,
        target_y=rhob_sh,
        value_col='GR'
    )

    if gr_ma is None or gr_sh is None:
        raise ValueError(
            "Tidak dapat menemukan titik terdekat untuk menentukan GR Matrix atau GR Shale.")

    return {
        "gr_ma": round(gr_ma, 4),
        "gr_sh": round(gr_sh, 4)
    }

