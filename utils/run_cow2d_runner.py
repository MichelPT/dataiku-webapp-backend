
import sys, time, math, argparse
from pathlib import Path

# Try to import patched modules first; fall back to originals.
def import_modules():
    cow = None
    cow2d = None
    try:
        import cow_fixed as cow  # patched
        import sys as _sys
        _sys.modules.setdefault("cow", cow)  # so cow2d_fixed can "import cow"
    except Exception:
        try:
            import cow as cow
        except Exception as e:
            raise ImportError("Neither cow_fixed nor cow could be imported: %r" % e)

    try:
        import cow2d_fixed as cow2d  # patched
    except Exception:
        try:
            import cow2d as cow2d
        except Exception as e:
            raise ImportError("Neither cow2d_fixed nor cow2d could be imported: %r" % e)
    return cow, cow2d

def read_signal_txt(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        vals = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                # ignore non-numeric lines
                pass
        if not vals:
            raise ValueError(f"No numeric data found in {path}")
        return vals

def gen_synthetic(n=200, shift=8, noise=0.02):
    import numpy as np
    x = np.linspace(0, 6.0, n)
    base = (
        0.7 * np.exp(-0.5*((x-1.5)/0.25)**2)
        + 1.0 * np.exp(-0.5*((x-3.0)/0.35)**2)
        + 0.6 * np.exp(-0.5*((x-4.2)/0.20)**2)
    )
    a1 = (base + noise*np.random.randn(n)).tolist()
    # warp/shift + slight scale to create a2
    x2 = np.linspace(0, 6.0, n)
    x2 = np.roll(x2, shift)
    base2 = (
        0.7 * np.exp(-0.5*((x2-1.5)/0.25)**2)
        + 1.0 * np.exp(-0.5*((x2-3.0)/0.35)**2)
        + 0.6 * np.exp(-0.5*((x2-4.2)/0.20)**2)
    )
    a2 = (1.03*base2 + noise*np.random.randn(n)).tolist()
    return a1, a2

def plot_matrix(mat, title, out_png):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    nrows = len(mat)
    ncols = len(mat[0])
    fig = plt.figure(title)
    ax = fig.add_subplot(111, projection='3d')
    yr = list(range(ncols))
    for i in range(nrows):
        z = mat[i]
        # plot each row as a curve at z=i
        ax.plot(yr, z, zs=i, zdir='z')
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_zlabel("Row")
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

def corr(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        m = min(a.size, b.size)
        a = a[:m]
        b = b[:m]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0,1])

def run(a1_path=None, a2_path=None, seg1=16, seg2=16, win1=6, win2=6, slack1=1, slack2=1, replicate_pairs=10, outdir=Path(".")):
    cow, cow2d = import_modules()

    # load signals or synthesize
    if a1_path and a2_path and Path(a1_path).exists() and Path(a2_path).exists():
        a1 = read_signal_txt(Path(a1_path))
        a2 = read_signal_txt(Path(a2_path))
        # If very long, crop like original test (9700:9900) but only if long enough
        if len(a1) >= 9900 and len(a2) >= 9900:
            a1 = a1[9700:9900]
            a2 = a2[9700:9900]
    else:
        a1, a2 = gen_synthetic(n=200, shift=10, noise=0.03)

    # Build Y with replicated pairs
    Y = []
    for _ in range(int(replicate_pairs)):
        Y.append(a1)
        Y.append(a2)

    # Construct and align
    c2 = cow2d.COW2D(Y, int(seg1), int(seg2), int(win1), int(win2), int(slack1), int(slack2))

    t0 = time.time()
    c2.align()
    t1 = time.time()

    # Outputs
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save plots
    before_png = outdir / "cow2d_before.png"
    after_png  = outdir / "cow2d_after.png"
    plot_matrix(c2.sample, "Before Alignment", str(before_png))
    plot_matrix(c2.target, "After Alignment", str(after_png))

    # Save aligned target as CSV
    import csv
    aligned_csv = outdir / "aligned_target.csv"
    with aligned_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in c2.target:
            w.writerow(row)

    # Correlation summary for each pair (0,1), (2,3), ...
    import numpy as np
    pairs = []
    for i in range(0, len(c2.sample), 2):
        if i+1 >= len(c2.sample): break
        b = corr(c2.sample[i], c2.sample[i+1])
        a = corr(c2.target[i], c2.target[i+1])
        pairs.append((i//2, b, a, (a-b) if (not math.isnan(b) and not math.isnan(a)) else float("nan")))

    summary_csv = outdir / "corr_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "corr_before", "corr_after", "delta"])
        for row in pairs:
            w.writerow(row)

    # Write a small text report
    report_txt = outdir / "run_report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("COW2D Runner Report\n")
        f.write("===================\n")
        f.write(f"Signals: {'files' if a1_path and a2_path else 'synthetic'}\n")
        if a1_path and a2_path:
            f.write(f"A1: {a1_path}\nA2: {a2_path}\n")
        f.write(f"Parameters: seg1={seg1}, seg2={seg2}, win1={win1}, win2={win2}, slack1={slack1}, slack2={slack2}\n")
        f.write(f"Pairs replicated: {replicate_pairs}\n")
        f.write(f"Align time: {t1 - t0:.4f} s\n")
        if pairs:
            import numpy as np
            arr = np.array([[r[1], r[2], r[3]] for r in pairs], dtype=float)
            mean_before = float(np.nanmean(arr[:,0]))
            mean_after  = float(np.nanmean(arr[:,1]))
            mean_delta  = float(np.nanmean(arr[:,2]))
            f.write(f"Mean corr (before): {mean_before:.4f}\n")
            f.write(f"Mean corr (after ): {mean_after:.4f}\n")
            f.write(f"Mean delta        : {mean_delta:.4f}\n")

    return {
        "before_png": str(before_png),
        "after_png": str(after_png),
        "aligned_csv": str(aligned_csv),
        "summary_csv": str(summary_csv),
        "report_txt": str(report_txt),
        "elapsed_sec": (t1 - t0)
    }

def main():
    p = argparse.ArgumentParser(description="Run 2D COW alignment (patched for Python 3).")
    p.add_argument("--a1", type=str, default="", help="Path to A1.txt")
    p.add_argument("--a2", type=str, default="", help="Path to A2.txt")
    p.add_argument("--seg1", type=int, default=16)
    p.add_argument("--seg2", type=int, default=16)
    p.add_argument("--win1", type=int, default=6)
    p.add_argument("--win2", type=int, default=6)
    p.add_argument("--slack1", type=int, default=1)
    p.add_argument("--slack2", type=int, default=1)
    p.add_argument("--replicate_pairs", type=int, default=10)
    p.add_argument("--outdir", type=str, default=".")
    args = p.parse_args()

    res = run(
        a1_path=args.a1 or None,
        a2_path=args.a2 or None,
        seg1=args.seg1, seg2=args.seg2,
        win1=args.win1, win2=args.win2,
        slack1=args.slack1, slack2=args.slack2,
        replicate_pairs=args.replicate_pairs,
        outdir=Path(args.outdir)
    )
    print("Outputs:")
    for k, v in res.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
