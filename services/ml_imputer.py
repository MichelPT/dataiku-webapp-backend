import pandas as pd
import os
import numpy as np  # Tambahkan ini
from math import radians
from sklearn.model_selection import train_test_split  # Tambahkan ini
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Tambahkan ini
from sklearn.metrics.pairwise import haversine_distances
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# FUNGSI UNTUK MENCARI SUMUR TETANGGA
# ==============================================================================


def haversine_dist(lat1, long1, lat2, long2):
    """Menghitung jarak Haversine antara dua titik koordinat."""
    origin = [lat1, long1]
    dest = [lat2, long2]
    origin_rad = [radians(_) for _ in origin]
    dest_rad = [radians(_) for _ in dest]
    # Jarak dalam KM
    dist = (haversine_distances([origin_rad, dest_rad]) * 6371)
    return dist[0, 1]


def rank_dist(ref_well_name, all_wells_df, well_count):
    """Mengurutkan sumur berdasarkan jarak dari sumur referensi dan mengembalikan N terdekat."""
    try:
        ref_row = all_wells_df[all_wells_df['WELL_NAME']
                               == ref_well_name].iloc[0]
        ref_lat = float(ref_row['LAT'])
        ref_long = float(ref_row['LONG'])

        # Buat dataframe target tanpa sumur referensi itu sendiri
        target_df = all_wells_df[all_wells_df['WELL_NAME']
                                 != ref_well_name].copy()

        dists = target_df.apply(
            lambda row: haversine_dist(
                ref_lat, ref_long, float(row['LAT']), float(row['LONG'])),
            axis=1
        )

        target_df['DISTANCE_KM'] = dists
        # Mengambil N sumur terdekat
        nearest_wells = target_df.sort_values('DISTANCE_KM').head(well_count)
        return nearest_wells['WELL_NAME'].to_list()
    except (IndexError, KeyError):
        # Jika sumur referensi tidak ditemukan di coord.csv
        return []

# ==============================================================================
# FUNGSI UNTUK MACHINE LEARNING (TRAINING & PREDICTION)
# ==============================================================================


# def train_rf_models(df, n_est=50, random_state=1337):
#     """Melatih model Random Forest untuk setiap kolom di DataFrame."""
#     models = {}
#     if 'DEPTH' not in df.columns:
#         raise ValueError("Kolom 'DEPTH' tidak ditemukan dalam data training!")

#     for target_col in df.columns:
#         # Hanya latih model untuk kolom numerik dan bukan DEPTH itu sendiri
#         if pd.api.types.is_numeric_dtype(df[target_col]) and target_col != 'DEPTH':

#             # Fitur adalah semua kolom numerik lain + DEPTH
#             features = [col for col in df.columns if pd.api.types.is_numeric_dtype(
#                 df[col]) and col != target_col]

#             # Siapkan data training, hapus baris dengan NaN di target atau fitur
#             train_data = df[[target_col] + features].dropna()

#             if len(train_data) < 10:  # Butuh data yang cukup untuk training
#                 continue

#             X = train_data[features]
#             y = train_data[target_col]

#             model = RandomForestRegressor(
#                 n_estimators=n_est, random_state=random_state, n_jobs=-1)
#             model.fit(X, y)

#             models[target_col] = {'model': model, 'features': features}

#     return models

def train_rf_models(df, available_features, n_est=50, random_state=1337):
    """
    Melatih model Random Forest untuk setiap kolom,
    sekaligus mengevaluasi kinerjanya dengan train-test split.
    """
    models = {}
    if 'DEPTH' not in df.columns:
        raise ValueError("Kolom 'DEPTH' tidak ditemukan dalam data training!")

    for target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]) and target_col != 'DEPTH':
            features = [f for f in available_features if f != target_col]
            train_data = df[[target_col] + features].dropna()

            if len(train_data) < 50:
                continue

            X = train_data[features]
            y = train_data[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            model = RandomForestRegressor(
                n_estimators=n_est, random_state=random_state, n_jobs=-1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            models[target_col] = {
                'model': model,
                'features': features,
                'metrics': {
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse
                }
            }

    return models


def predict_missing_values(df, models, logs_to_fill):
    """
    Memprediksi nilai yang hilang. Fitur dijamin sudah cocok.
    """
    df_filled = df.copy()

    for log in logs_to_fill:
        if log not in models:
            print(
                f"Peringatan: Tidak ada model yang terlatih untuk log '{log}'. Melewati...")
            continue

        model_info = models[log]
        model = model_info['model']
        # Fitur dari model sekarang dijamin ada di dataframe 'df'
        features = model_info['features']

        mask_missing = df_filled[log].isna(
        ) & df_filled[features].notna().all(axis=1)

        if not mask_missing.any():
            continue

        X_to_predict = df_filled.loc[mask_missing, features]

        if X_to_predict.empty:
            continue

        new_col_name = f"{log}_FM"
        df_filled[new_col_name] = df_filled[log]

        predicted_values = model.predict(X_to_predict)
        df_filled.loc[mask_missing, new_col_name] = predicted_values

    return df_filled

# ==============================================================================
# FUNGSI UTAMA (ORCHESTRATOR)
# ==============================================================================


def run_ml_imputation_for_well(target_well_path, logs_to_fill, coord_df, well_count=5):
    """
    Menjalankan seluruh proses imputasi ML untuk satu sumur target.
    1. Cari tetangga terdekat.
    2. Muat data tetangga sebagai data training.
    3. Latih model.
    4. Lakukan prediksi pada sumur target.
    """
    print(
        f"Memulai proses imputasi ML untuk: {os.path.basename(target_well_path)}")

    # Muat data sumur target
    df_target = pd.read_csv(target_well_path)
    target_well_name = os.path.basename(target_well_path).replace('.csv', '')

    # Cari sumur tetangga terdekat
    neighbor_wells = rank_dist(target_well_name, coord_df, well_count)
    if not neighbor_wells:
        print(
            f"Peringatan: Tidak ditemukan tetangga untuk {target_well_name}. Imputasi ML dibatalkan.")
        return df_target  # Kembalikan dataframe asli jika tidak ada tetangga

    print(f"Sumur tetangga ditemukan: {neighbor_wells}")

    # Muat data training dari sumur tetangga
    base_dir = os.path.dirname(target_well_path)
    training_dfs = []
    for well in neighbor_wells:
        neighbor_path = os.path.join(base_dir, f"{well}.csv")
        if os.path.exists(neighbor_path):
            training_dfs.append(pd.read_csv(neighbor_path))

    if not training_dfs:
        print("Peringatan: Gagal memuat data dari sumur tetangga. Imputasi ML dibatalkan.")
        return df_target

    df_training = pd.concat(training_dfs, ignore_index=True)

    train_cols = df_training.columns
    target_cols = df_target.columns
    common_features = train_cols.intersection(target_cols).tolist()

    # Pastikan hanya fitur numerik yang dipakai
    common_numeric_features = [
        col for col in common_features if pd.api.types.is_numeric_dtype(df_training[col])
    ]
    print(
        f"Fitur yang akan digunakan untuk training & prediksi: {common_numeric_features}")

    print("Melatih model Random Forest...")
    models = train_rf_models(df_training, common_numeric_features)

    print("\n--- Hasil Evaluasi Kinerja Model (dari data tetangga) ---")
    if not models:
        print("Tidak ada model yang berhasil dilatih dan dievaluasi.")
    else:
        for log, data in models.items():
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"  Log: {log}")
                print(f"    R-squared: {metrics['r2']:.4f}")
                print(f"    MAE      : {metrics['mae']:.4f}")
                print(f"    RMSE     : {metrics['rmse']:.4f}")
    print("----------------------------------------------------------\n")

    if not models:
        print("Peringatan: Gagal melatih model. Imputasi ML dibatalkan.")
        return df_target

    # 5. Lakukan prediksi
    print("Melakukan prediksi untuk mengisi nilai yang hilang...")
    df_imputed = predict_missing_values(df_target, models, logs_to_fill)

    print(f"Proses imputasi ML untuk {target_well_name} selesai.")
    return df_imputed
