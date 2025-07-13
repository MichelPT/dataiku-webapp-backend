import lasio
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def normalize(series):
    std_dev = np.std(series)
    if std_dev == 0:
        # Jika semua nilai sama, kembalikan array berisi nol
        return np.zeros_like(series)
    return (series - np.mean(series)) / std_dev


def depth_matching(ref_las_path: str, lwd_las_path: str, num_chunks: int = 10):
    """
    Menjalankan logika DTW dan mengembalikan tiga DataFrame: 
    data referensi, data LWD asli, dan data hasil alignment.
    """
    try:
        ref_las = lasio.read(ref_las_path)
        lwd_las = lasio.read(lwd_las_path)

        ref_df = ref_las.df().reset_index()[["DEPTH", "GR_CAL"]].dropna()
        ref_df.columns = ["Depth", "GR"]

        lwd_df = lwd_las.df().reset_index()[["DEPTH", "DGRCC"]].dropna()
        lwd_df.columns = ["Depth", "DGRCC"]

        N_ref = len(ref_df)
        N_lwd = len(lwd_df)

        ref_chunk_size = N_ref // num_chunks
        lwd_chunk_size = N_lwd // num_chunks

        all_chunks = []

        for i in range(num_chunks):
            ref_start = i * ref_chunk_size
            ref_end = N_ref if i == num_chunks - \
                1 else (i + 1) * ref_chunk_size
            ref_chunk = ref_df.iloc[ref_start:ref_end]

            lwd_start = i * lwd_chunk_size
            lwd_end = N_lwd if i == num_chunks - \
                1 else (i + 1) * lwd_chunk_size
            lwd_chunk = lwd_df.iloc[lwd_start:lwd_end]

            if len(ref_chunk) < 2 or len(lwd_chunk) < 2:
                continue

            ref_signal = normalize(ref_chunk["GR"].values)
            lwd_signal = normalize(lwd_chunk["DGRCC"].values)

            path = dtw.warping_path(lwd_signal, ref_signal)

            aligned_depths = []
            aligned_dgrcc = []
            for lwd_idx, ref_idx in path:
                aligned_depths.append(ref_chunk.iloc[ref_idx]["Depth"])
                aligned_dgrcc.append(lwd_chunk.iloc[lwd_idx]["DGRCC"])

            aligned_df = pd.DataFrame({
                "Depth": aligned_depths,
                "Aligned_DGRCC": aligned_dgrcc
            }).sort_values(by="Depth")

            interp_func = interp1d(aligned_df["Depth"], aligned_df["Aligned_DGRCC"],
                                   kind='linear', bounds_error=False, fill_value="extrapolate")
            interp_dgrcc = interp_func(ref_chunk["Depth"].values)

            chunk_result = pd.DataFrame({
                "Depth": ref_chunk["Depth"].values,
                "REF_GR": ref_chunk["GR"].values,
                "LWD_DGRCC_Aligned": interp_dgrcc
            })

            all_chunks.append(chunk_result)

            final_df = pd.concat(all_chunks, ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset="Depth").sort_values(by="Depth")

    except Exception as e:
        print(f"Error di dalam depth_matching_logic: {e}")
        return None, None, None

    return ref_df, lwd_df, final_df


def plot_depth_matching_results(ref_df, lwd_df, final_df):
    """
    Menerima 3 DataFrame dan membuat plot 4-panel yang komprehensif.
    """
    if ref_df is None or lwd_df is None or final_df is None:
        raise ValueError("Data input untuk plotting tidak boleh None.")

    # Buat 4 subplot dengan sumbu Y yang sama
    fig = make_subplots(
        rows=1, cols=4,
        shared_yaxes=True,
        subplot_titles=("Reference Log", "LWD Log (Original)",
                        "Before Alignment", "After Alignment (DTW)")
    )

    # --- Panel 1 (Paling Kiri): Reference Log ---
    fig.add_trace(go.Scattergl(
        x=ref_df["GR"], y=ref_df["Depth"], name='REF GR',
        line=dict(color='black')
    ), row=1, col=1)

    # --- Panel 2: LWD Log Asli ---
    fig.add_trace(go.Scattergl(
        x=lwd_df["DGRCC"], y=lwd_df["Depth"], name='LWD GR',
        line=dict(color='red')
    ), row=1, col=2)

    # --- Panel 3: Sebelum Alignment (Ditumpuk) ---
    fig.add_trace(go.Scattergl(
        x=ref_df["GR"], y=ref_df["Depth"], name='REF GR (Before)',
        line=dict(color='black'), legendgroup='before'
    ), row=1, col=3)
    fig.add_trace(go.Scattergl(
        x=lwd_df["DGRCC"], y=lwd_df["Depth"], name='LWD GR (Before)',
        line=dict(color='red', dash='dash'), legendgroup='before'
    ), row=1, col=3)

    # --- Panel 4 (Paling Kanan): Setelah Alignment (Ditumpuk) ---
    fig.add_trace(go.Scattergl(
        x=final_df["REF_GR"], y=final_df["Depth"], name='REF GR (After)',
        line=dict(color='black'), legendgroup='after'
    ), row=1, col=4)
    fig.add_trace(go.Scattergl(
        x=final_df["LWD_DGRCC_Aligned"], y=final_df[
            "Depth"], name='LWD Aligned (After)',
        line=dict(color='red', dash='dash'), legendgroup='after'
    ), row=1, col=4)

    fig.update_layout(
        title_text="Depth Matching Analysis",
        height=8600,
        showlegend=False,
        template="plotly_white",
        yaxis=dict(autorange='reversed', title_text="Depth (m)"),
        hovermode="y unified",
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='red',
        dtick=15,
        griddash='dot',
        showline=True,
        linewidth=1.5,
        linecolor='black',
        mirror=True
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor='black',
        mirror=True
    )

    return fig
