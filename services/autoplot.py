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
